import time
from math import comb, factorial
from typing import Union, Optional
import numpy as np
from src.middlewares.slogger import SafeLogger
from src.funcs.base import emd_efecto, ABECEDARY, LOWER_ABECEDARY, seleccionar_subestado
from src.middlewares.profile import profiler_manager, profile
from src.controllers.manager import Manager
from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.constants.models import (
    KGEOMIP_ANALYSIS_TAG,
    KGEOMIP_LABEL,
    KGEOMIP_STRATEGY_TAG,
)
from src.constants.base import (
    TYPE_TAG,
    NET_LABEL,
    INFTY_POS,
    ACTUAL,
    EFECTO,
    VOID_STR,
)
from src.models.base.application import aplicacion
from src.models.enums.notation import Notation


# =============================================================================
# Funciones auxiliares para enumeración de k-particiones
# =============================================================================


def stirling2(n: int, k: int) -> int:
    """Número de Stirling de segunda especie S(n,k).

    Cuenta las formas de particionar un conjunto de n elementos
    etiquetados en k subconjuntos no vacíos.

    Args:
        n: Tamaño del conjunto.
        k: Número de bloques.

    Returns:
        S(n,k) como entero.
    """
    if k > n or k < 1:
        return 0
    if k == 1 or k == n:
        return 1
    total = 0
    for i in range(k + 1):
        term = comb(k, i) * ((k - i) ** n)
        total += -term if i % 2 else term
    return total // factorial(k)


def _rgs_generator(m: int, k: int):
    """Genera todas las restricted growth strings de longitud m con k bloques.

    Cada RGS a[0..m-1] cumple:
      - a[0] = 0
      - a[j+1] <= 1 + max(a[0..j])
      - max(a) = k-1

    Args:
        m: Longitud de la cadena (número de vértices).
        k: Número de bloques.

    Yields:
        List[int] con la asignación de cada vértice a un grupo.
    """
    def _recurse(prefix: list[int], max_so_far: int):
        if len(prefix) == m:
            if max_so_far == k - 1:
                yield prefix
            return
        upper = min(max_so_far + 2, k)
        for nxt in range(upper):
            yield from _recurse(prefix + [nxt], max(max_so_far, nxt))
    yield from _recurse([], -1)


def _rgs_to_grupos(rgs: list[int], vertices: list) -> list[list]:
    """Convierte restricted growth string en lista de k grupos de vértices.

    Args:
        rgs: Restricted growth string (asignación grupo por vértice).
        vertices: Lista de vértices paralela a rgs.

    Returns:
        Lista de k listas, cada una con los vértices de ese grupo.
    """
    k = max(rgs) + 1
    grupos: list[list] = [[] for _ in range(k)]
    for idx, g in enumerate(rgs):
        grupos[g].append(vertices[idx])
    return grupos


def fmt_kparte_q(groups: list[list[tuple[int, int]]]) -> str:
    """Formatea una k-partición como |purview..|..| / |mechanism..|..|.

    Cada grupo se renderiza como una columna vertical:
      línea superior: efecto (mayúsculas)
      línea inferior: mecanismo (minúsculas)

    Args:
        groups: Lista de k grupos, cada grupo con tuplas (tiempo, índice).

    Returns:
        Cadena formateada con dos filas (purview arriba, mechanism abajo).
    """
    purv_rows: list[str] = []
    mech_rows: list[str] = []
    for group in groups:
        purv: list[int] = []
        mech: list[int] = []
        for t, idx in sorted(group, key=lambda x: x[1]):
            (purv if t == EFECTO else mech).append(idx)
        str_purv = ",".join(ABECEDARY[i] for i in purv) if purv else VOID_STR
        str_mech = ",".join(LOWER_ABECEDARY[i] for i in mech) if mech else VOID_STR
        width = max(len(str_purv), len(str_mech)) + 2
        purv_rows.append(f"|{str_purv:^{width}}|")
        mech_rows.append(f"|{str_mech:^{width}}|")
    return "".join(purv_rows) + "\n" + "".join(mech_rows)


# =============================================================================
# Clase principal: KGeoMIP
# =============================================================================


class KGeoMIP(SIA):
    """Estrategia híbrida para la k-partición de mínima información (k-MIP).

    Extiende el enfoque GeoMIP para identificar la k-partición que minimiza
    la pérdida EMD para k ∈ {2,3,4,5}. Representa el espacio de estados
    como un hipercubo n-dimensional donde cada hiperplano de separación
    define una frontera entre grupos de la partición.

    La tabla de costos de transiciones (distribuciones de probabilidad
    aplanadas por nodo) se precomputa una sola vez en ``_flat_data`` y se
    reutiliza en todas las evaluaciones de k-particiones, evitando
    recalcular marginalizaciones redundantes.

    Modo de búsqueda:
      - **Exhaustivo** (S(2n,k) ≤ 1e5, o forzado): enumera todas las
        k-particiones mediante restricted growth strings y selecciona la
        de mínima pérdida EMD. Garantiza la k-MIP óptima global.
      - **Geométrico-heurístico** (S(2n,k) > 1e5, o forzado): construye
        la partición jerárquicamente usando k-1 hiperplanos. Parte de la
        bipartición óptima (algoritmo geométrico original) y subdivide
        iterativamente el grupo cuya división interna más reduce la
        pérdida EMD.

    Para k=2 el resultado es idéntico al de GeometricSIA.

    Args:
        gestor: Manager con la configuración del sistema.
        k: Número de grupos (2 ≤ k ≤ 5).
        busqueda_exhaustiva: Si True fuerza modo exhaustivo;
            si False fuerza modo heurístico;
            si None (default) selección automática según S(2n,k).

    Attributes:
        _flat_data: Lista de arrays de probabilidad aplanados por nodo
            (precomputados una vez y reutilizados).
        memoria_particiones: Diccionario clave -> (EMD, distribución)
            para particiones ya evaluadas.
    """

    THRESHOLD_EXHAUSTIVE: int = 100_000

    def __init__(
        self,
        gestor: Manager,
        k: int = 2,
        busqueda_exhaustiva: Optional[bool] = None,
    ):
        super().__init__(gestor)
        self.k = k
        self.busqueda_exhaustiva = busqueda_exhaustiva
        profiler_manager.start_session(
            f"{NET_LABEL}{len(gestor.estado_inicial)}{gestor.pagina}"
        )
        self.logger = SafeLogger(KGEOMIP_STRATEGY_TAG)
        self.vertices: set[tuple] = set()
        self.memoria_particiones: dict = {}
        self._flat_data: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Método principal
    # ------------------------------------------------------------------

    @profile(context={TYPE_TAG: KGEOMIP_ANALYSIS_TAG})
    def aplicar_estrategia(
        self,
        condicion: str,
        alcance: str,
        mecanismo: str,
        tpm: np.ndarray,
    ):
        """Ejecuta la estrategia KGeoMIP para encontrar la k-MIP.

        Args:
            condicion: Cadena de bits para condiciones de fondo.
            alcance: Cadena de bits para substracción del alcance.
            mecanismo: Cadena de bits para substracción del mecanismo.
            tpm: Matriz de probabilidad de transición cargada.

        Returns:
            Solution con la mejor k-partición encontrada.
        """
        if self.k < 2 or self.k > 5:
            raise ValueError(
                f"k debe estar entre 2 y 5, recibido {self.k}"
            )

        self.sia_preparar_subsistema(condicion, alcance, mecanismo, tpm)
        self._tpm = tpm

        for ncubo in self.sia_subsistema.ncubos:
            self._flat_data.append(ncubo.data.ravel())

        futuro = tuple(
            (EFECTO, i) for i in self.sia_subsistema.indices_ncubos
        )
        presente = tuple(
            (ACTUAL, i) for i in self.sia_subsistema.dims_ncubos
        )
        vertices = list(presente + futuro)
        self.vertices = set(vertices)

        self.logger.critic(
            f"Iniciando KGeoMIP k={self.k} con {len(vertices)} vértices"
        )

        if self.k == 2:
            groups, perdida, dist_part = self._resolver_biparticion(vertices)
        else:
            groups, perdida, dist_part = self.find_k_mip(vertices, self.k)

        return Solution(
            estrategia=KGEOMIP_LABEL,
            perdida=perdida,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=dist_part,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_kparte_q(groups),
        )

    # ------------------------------------------------------------------
    # Delegación k=2 → GeometricSIA
    # ------------------------------------------------------------------

    def _resolver_biparticion(self, vertices):
        """Resuelve k=2 delegando al algoritmo geométrico original.

        Construye un objeto GeometricSIA liviano reutilizando los datos
        precomputados del subsistema, garantizando resultados idénticos
        a la ejecución directa de GeometricSIA.

        Args:
            vertices: Lista de vértices del subsistema.

        Returns:
            Tuple (groups, perdida, dist_particion) con groups teniendo
            exactamente 2 grupos.
        """
        from src.controllers.strategies.geometric import GeometricSIA
        geo = GeometricSIA.__new__(GeometricSIA)
        geo.sia_gestor = self.sia_gestor
        geo.sia_logger = self.sia_logger
        geo.sia_subsistema = self.sia_subsistema
        geo.sia_dists_marginales = self.sia_dists_marginales
        geo.sia_tiempo_inicio = self.sia_tiempo_inicio
        geo.etiquetas = [tuple(s.lower() for s in ABECEDARY), ABECEDARY]
        geo.logger = self.logger
        geo.tabla_transiciones = {}
        geo.vertices = self.vertices
        geo.tabla = {}
        geo.memoria_particiones = {}
        geo._flat_data = self._flat_data

        dims = self.sia_subsistema.dims_ncubos
        geo.estado_inicial = self.sia_subsistema.estado_inicial[dims]
        geo.estado_final = 1 - geo.estado_inicial
        geo.idx_ncubos = list(range(len(self.sia_subsistema.indices_ncubos)))

        mip = geo.find_mip()
        emd_val, dist_val = geo.memoria_particiones[mip]

        mip_set = set(mip)
        group1 = list(mip_set)
        group2 = list(self.vertices - mip_set)
        groups = [group1, group2]

        return groups, emd_val, dist_val

    # ------------------------------------------------------------------
    # Selección de modo
    # ------------------------------------------------------------------

    def _usar_exhaustivo(self, num_vertices: int) -> bool:
        """Determina si usar búsqueda exhaustiva según S(2n,k).

        Args:
            num_vertices: Número total de vértices (2n).

        Returns:
            True si debe usarse modo exhaustivo.
        """
        if self.busqueda_exhaustiva is True:
            return True
        if self.busqueda_exhaustiva is False:
            return False
        return stirling2(num_vertices, self.k) <= self.THRESHOLD_EXHAUSTIVE

    # ------------------------------------------------------------------
    # Orquestador k-MIP
    # ------------------------------------------------------------------

    def find_k_mip(
        self,
        vertices: list,
        k: int,
    ):
        """Encuentra la k-partición de mínima pérdida.

        Selecciona automáticamente modo exhaustivo o heurístico según
        el tamaño del sistema.

        Args:
            vertices: Lista de vértices a particionar.
            k: Número de grupos deseados.

        Returns:
            Tuple (groups, perdida, dist_particion).
        """
        num_vertices = len(vertices)
        if self._usar_exhaustivo(num_vertices):
            self.logger.critic(
                f"Modo exhaustivo: S({num_vertices},{k}) = "
                f"{stirling2(num_vertices, k)}"
            )
            return self._exhaustivo(vertices, k)

        self.logger.critic(
            f"Modo heurístico (k-1 hiperplanos): S({num_vertices},{k}) = "
            f"{stirling2(num_vertices, k)}"
        )
        return self._greedy(vertices, k)

    # ------------------------------------------------------------------
    # Modo exhaustivo
    # ------------------------------------------------------------------

    def _exhaustivo(
        self,
        vertices: list,
        k: int,
    ):
        """Modo exhaustivo: evalúa todas las k-particiones posibles.

        Genera exactamente S(|vertices|,k) particiones mediante
        restricted growth strings y selecciona la de mínima EMD.
        Garantiza la k-MIP óptima global.

        Adecuado cuando S(2n,k) ≤ 1e5 (típicamente n ≤ 5 para todo k,
        n ≤ 6 para k=3).

        Los hiperplanos de separación se exploran implícitamente:
        cada RGS asigna cada vértice a uno de k grupos, donde la
        frontera entre grupos i y j corresponde al hiperplano que
        separa los vértices asignados a cada grupo.

        Args:
            vertices: Lista de vértices a particionar.
            k: Número de grupos.

        Returns:
            Tuple (groups, perdida, dist_particion).
        """
        best_emd = float("inf")
        best_groups = None
        best_dist = None

        total = stirling2(len(vertices), k)
        evaluated = 0

        for rgs in _rgs_generator(len(vertices), k):
            groups = _rgs_to_grupos(rgs, vertices)
            emd, dist = self.evaluate_k_partition(groups)
            evaluated += 1
            if emd < best_emd:
                best_emd = emd
                best_groups = groups
                best_dist = dist

        self.logger.critic(
            f"Exhaustivo: evaluadas {evaluated}/{total} particiones, "
            f"mejor EMD = {best_emd:.6f}"
        )
        return best_groups, best_emd, best_dist

    # ------------------------------------------------------------------
    # Modo heurístico (k-1 hiperplanos)
    # ------------------------------------------------------------------

    def _greedy(
        self,
        vertices: list,
        k: int,
    ):
        """Modo heurístico basado en k-1 hiperplanos jerárquicos.

        Estrategia:
          1. Encuentra el primer hiperplano (bipartición óptima) usando
             el algoritmo geométrico original.
          2. Para i = 2 .. k-1:
             - Encuentra el hiperplano i-ésimo subdividiendo el grupo
               cuya partición interna más reduce la pérdida EMD global.
          3. Cada hiperplano se define por el conjunto de mecanismos
             que un subgrupo de efectos conserva.

        Este enfoque es escalable a sistemas mayores (S(2n,k) > 1e5)
        pero no garantiza optimalidad global para k>2.

        Args:
            vertices: Lista de vértices a particionar.
            k: Número de grupos deseados.

        Returns:
            Tuple (groups, perdida, dist_particion).
        """
        mip = self._optimal_bipartition(vertices)
        mip_set = set(mip)
        groups = [list(mip_set), list(set(vertices) - mip_set)]
        best_emd, best_dist = self.evaluate_k_partition(groups)

        self.logger.critic(
            f"Hiperplano 1: EMD={best_emd:.6f}, "
            f"grupos=[{len(groups[0])},{len(groups[1])}] vértices"
        )

        while len(groups) < k:
            best_idx = -1
            best_split = None
            best_candidate_emd = float("inf")
            best_candidate_dist = None

            # Evaluar cada grupo existente para posible subdivisión
            for idx, group in enumerate(groups):
                if len(group) < 2:
                    continue
                splits = self._enumerate_splits(group)
                for g1, g2 in splits:
                    candidate = (
                        groups[:idx] + groups[idx + 1:] + [g1, g2]
                    )
                    emd_val, dist_val = self.evaluate_k_partition(candidate)
                    if emd_val < best_candidate_emd:
                        best_candidate_emd = emd_val
                        best_candidate_dist = dist_val
                        best_idx = idx
                        best_split = (g1, g2)

            if best_idx == -1:
                self.logger.critic(
                    "No se encontró grupo divisible — terminando "
                    f"con {len(groups)} grupos"
                )
                break

            groups = (
                groups[:best_idx]
                + groups[best_idx + 1:]
                + [list(best_split[0]), list(best_split[1])]
            )
            best_emd = best_candidate_emd
            best_dist = best_candidate_dist

            self.logger.critic(
                f"Hiperplano {len(groups)-1}: subdividido grupo {best_idx}, "
                f"EMD={best_emd:.6f}"
            )

        return groups, best_emd, best_dist

    SPLIT_EXHAUSTIVE_LIMIT: int = 8
    SPLIT_RANDOM_SAMPLES: int = 50

    def _enumerate_splits(
        self,
        group: list[tuple[int, int]],
    ):
        """Genera candidatos de bipartición para un grupo.

        Para grupos pequeños (≤ SPLIT_EXHAUSTIVE_LIMIT) enumera todas
        las 2^(n-1)-1 biparticiones no triviales.

        Para grupos grandes genera SPLIT_RANDOM_SAMPLES biparticiones
        aleatorias (cada vértice se asigna con p=0.5 a cada lado).

        Args:
            group: Lista de vértices a dividir.

        Returns:
            Lista de tuplas (g1, g2) con las biparticiones candidatas.
        """
        if len(group) < 2:
            return []
        if len(group) == 2:
            return [([group[0]], [group[1]])]
        n = len(group)
        if n <= self.SPLIT_EXHAUSTIVE_LIMIT:
            splits = []
            for mask in range(1, 1 << (n - 1)):
                g1, g2 = [], []
                for i in range(n):
                    (g1 if (mask >> i) & 1 else g2).append(group[i])
                splits.append((g1, g2))
            return splits
        rng = np.random.default_rng(42)
        splits = []
        for _ in range(self.SPLIT_RANDOM_SAMPLES):
            g1, g2 = [], []
            for v in group:
                (g1 if rng.integers(2) else g2).append(v)
            if g1 and g2:
                splits.append((g1, g2))
        return splits

    def _optimal_bipartition(self, vertices):
        """Encuentra la bipartición óptima via algoritmo geométrico.

        Delega a GeometricSIA.find_mip() para obtener la bipartición
        de mínima pérdida del conjunto completo de vértices.

        Args:
            vertices: Lista completa de vértices del subsistema.

        Returns:
            Tupla de vértices que forman una de las dos partes de la
            bipartición óptima.
        """
        from src.controllers.strategies.geometric import GeometricSIA
        geo = GeometricSIA.__new__(GeometricSIA)
        geo.sia_gestor = self.sia_gestor
        geo.sia_logger = self.sia_logger
        geo.sia_subsistema = self.sia_subsistema
        geo.sia_dists_marginales = self.sia_dists_marginales
        geo.sia_tiempo_inicio = self.sia_tiempo_inicio
        geo.etiquetas = [tuple(s.lower() for s in ABECEDARY), ABECEDARY]
        geo.logger = self.logger
        geo.tabla_transiciones = {}
        geo.vertices = self.vertices
        geo.tabla = {}
        geo.memoria_particiones = {}
        geo._flat_data = self._flat_data

        dims = self.sia_subsistema.dims_ncubos
        geo.estado_inicial = self.sia_subsistema.estado_inicial[dims]
        geo.estado_final = 1 - geo.estado_inicial
        geo.idx_ncubos = list(range(len(self.sia_subsistema.indices_ncubos)))

        return geo.find_mip()

    # ------------------------------------------------------------------
    # Evaluación de una k-partición
    # ------------------------------------------------------------------

    def _causa_precomputar(self):
        """Precomputa likelihoods individuales y posterior no-particionado."""
        if hasattr(self, '_causa_cache'):
            return
        tpm = self._tpm
        estado = self.sia_subsistema.estado_inicial
        n_nodes = len(estado)
        n_states = 1 << n_nodes
        L = np.where(estado[np.newaxis, :] == 1, tpm, 1.0 - tpm)
        likelihood_all = np.prod(L[:, :], axis=1)
        Z_all = np.sum(likelihood_all)
        posterior_all = (
            likelihood_all / Z_all
            if Z_all > 0
            else np.ones(n_states, dtype=np.float64) / n_states
        )
        self._causa_cache = {
            'L': L,
            'posterior_all': posterior_all,
            'state_bits': np.arange(n_states),
        }

    def calculate_cause_repertoire(
        self,
        groups: list[list[tuple[int, int]]],
    ):
        """Calcula el repertorio causa para una k-partición usando la Regla de Bayes.

        Repertorio causa: P(pasado | mecanismo) usando:
          p(Causa | Efecto) = p(Efecto | Causa) × p_per(Causa) / p(Efecto)

        Donde:
          - Efecto = estado actual del mecanismo (tiempo t)
          - Causa = estado pasado del purview (tiempo t-1)
          - p(Efecto | Causa) desde la TPM: P(mecanismo | estado_pasado)
          - p_per(Causa) = 1/2^|purview| (distribución previa uniforme)
          - Renormalización obligatoria: las columnas no suman 1

        Args:
            groups: Lista de grupos, cada uno con tuplas (tiempo, índice).

        Returns:
            Tuple (cause_emd_loss, cause_partitioned_distribution).
        """
        self._causa_precomputar()
        cache = self._causa_cache
        L = cache['L']
        posterior_all = cache['posterior_all']
        state_bits = cache['state_bits']

        all_purv = sorted(set(
            idx for group in groups for t, idx in group if t == EFECTO
        ))

        purv_to_mech: dict[int, list[int]] = {}
        for group in groups:
            mech = sorted([idx for t, idx in group if t == ACTUAL])
            purv = sorted([idx for t, idx in group if t == EFECTO])
            for p in purv:
                purv_to_mech[p] = mech

        n_states = len(posterior_all)
        unpart = np.empty(len(all_purv), dtype=np.float64)
        for p_idx, p in enumerate(all_purv):
            mask = ((state_bits >> p) & 1) == 1
            unpart[p_idx] = np.sum(posterior_all[mask])

        part = np.empty(len(all_purv), dtype=np.float64)
        for p_idx, p in enumerate(all_purv):
            mech_for_p = purv_to_mech.get(p, [])
            if not mech_for_p:
                part[p_idx] = 0.5
                continue
            likelihood_g = np.prod(L[:, mech_for_p], axis=1)
            Z_g = np.sum(likelihood_g)
            posterior_g = (
                likelihood_g / Z_g
                if Z_g > 0
                else np.ones(n_states, dtype=np.float64) / n_states
            )
            mask = ((state_bits >> p) & 1) == 1
            part[p_idx] = np.sum(posterior_g[mask])

        cause_emd = float(np.sum(np.abs(unpart - part)))
        return cause_emd, part

    def evaluate_k_partition(
        self,
        groups: list[list[tuple[int, int]]],
    ):
        """Evalúa una k-partición calculando su pérdida EMD mínima
        entre los repertorios causa y efecto.

        La pérdida (φ) de un concepto es el mínimo entre la irreducibilidad
        de su repertorio causa y la de su repertorio efecto.

        Para cada grupo de la partición:
          - Las variables efecto (purview) marginalizan su cubo para
            conservar solo las dimensiones de mecanismo de su grupo.
          - Esto corresponde a proyectar el hipercubo n-dimensional
            sobre los subespacios definidos por k-1 hiperplanos: cada
            hiperplano separa las dimensiones que un subconjunto de
            nodos efecto conserva de las que descarta.
          - La distribución particionada es el vector de off-probabilities
            por nodo, donde cada nodo efecto solo "ve" las dimensiones
            de mecanismo de su grupo.

        La EMD se computa contra la distribución marginal del subsistema
        original (precomputada en sia_dists_marginales).

        Args:
            groups: Lista de grupos, cada uno con tuplas (tiempo, índice).

        Returns:
            Tuple (emd_loss, partition_distribution).
        """
        futuro_a_mecanismo: dict[int, list[int]] = {}
        for group in groups:
            mech = sorted([idx for t, idx in group if t == ACTUAL])
            purv = sorted([idx for t, idx in group if t == EFECTO])
            for p in purv:
                futuro_a_mecanismo[p] = mech

        n = len(self.sia_subsistema.ncubos)
        part_dist = np.empty(n, dtype=np.float32)

        for i, cube in enumerate(self.sia_subsistema.ncubos):
            if cube.indice in futuro_a_mecanismo:
                mech = futuro_a_mecanismo[cube.indice]
                if mech:
                    keep = np.array(mech, dtype=np.int8)
                    marginalize = np.setdiff1d(cube.dims, keep)
                    marginalized = cube.marginalizar(marginalize)
                else:
                    marginalized = cube.marginalizar(cube.dims)

                prob = marginalized.data
                if marginalized.dims.size:
                    sub_state = tuple(
                        int(self.sia_subsistema.estado_inicial[j])
                        for j in marginalized.dims
                    )
                    if aplicacion.notacion == Notation.LIL_ENDIAN.value:
                        sub_state = sub_state[::-1]
                    prob = marginalized.data[sub_state]
                part_dist[i] = 1 - float(prob)
            else:
                part_dist[i] = 0.0

        emd_efecto_val = emd_efecto(part_dist, self.sia_dists_marginales)
        emd_causa_val, _ = self.calculate_cause_repertoire(groups)

        return min(emd_efecto_val, emd_causa_val), part_dist

    # ------------------------------------------------------------------
    # Utilidad: complemento de nodos
    # ------------------------------------------------------------------

    def nodes_complement(self, nodes):
        """Retorna el complemento del conjunto respecto a todos los vértices.

        Args:
            nodes: Lista o conjunto de vértices.

        Returns:
            Lista de vértices no incluidos en nodes.
        """
        return list(set(self.vertices) - set(nodes))
