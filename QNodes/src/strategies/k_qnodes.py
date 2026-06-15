import time
from typing import Union
from math import comb, factorial
import numpy as np
from src.middlewares.slogger import SafeLogger
from src.funcs.iit import emd_efecto, ABECEDARY, LOWER_ABECEDARY
from src.middlewares.profile import gestor_perfilado, profile
from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.constants.base import (
    TYPE_TAG,
    COLS_IDX,
    NET_LABEL,
    INFTY_POS,
    LAST_IDX,
    EFFECT,
    ACTUAL,
    VOID_STR,
)
from src.models.base.application import aplicacion
from src.models.enums.notation import Notation


KQNODES_LABEL = "KQNodes"
KQNODES_STRATEGY_TAG = f"{KQNODES_LABEL}_strategy"
KQNODES_ANALYSIS_TAG = f"{KQNODES_LABEL}_analysis"




def stirling2(n: int, k: int) -> int:
    """Stirling number of the second kind S(n,k).


    Número de formas de particionar un conjunto de n elementos
    etiquetados en k subconjuntos no vacíos.
    """
    if k > n or k < 1:
        return 0
    if k == 1 or k == n:
        return 1
    total = 0
    for i in range(k + 1):
        term = comb(k, i) * ((k - i) ** n)
        if i % 2:
            total -= term
        else:
            total += term
    return total // factorial(k)




def _rgs_generator(m: int, k: int):
    """Genera todas las restricted growth strings de longitud m con exactamente k bloques.


    Cada RGS es una lista a[0..m-1] donde:
      - a[0] = 0
      - a[j+1] ≤ 1 + max(a[0..j])
      - max(a) = k-1  (exactamente k bloques)
    """
    def recurse(prefix: list[int], max_so_far: int):
        if len(prefix) == m:
            if max_so_far == k - 1:
                yield prefix
            return
        for nxt in range(max_so_far + 2):
            if nxt >= k:
                break
            yield from recurse(prefix + [nxt], max(max_so_far, nxt))


    yield from recurse([], -1)




def _rgs_to_grupos(rgs: list[int], vertices: list) -> list[list]:
    """Convierte una RGS en lista de k grupos de vértices."""
    k = max(rgs) + 1
    grupos: list[list] = [[] for _ in range(k)]
    for idx, g in enumerate(rgs):
        grupos[g].append(vertices[idx])
    return grupos




def fmt_kparte_q(groups: list[list[tuple[int, int]]]) -> str:
    purv_rows: list[str] = []
    mech_rows: list[str] = []
    for group in groups:
        purv: list[int] = []
        mech: list[int] = []
        for t, idx in sorted(group, key=lambda x: x[1]):
            (purv if t == EFFECT else mech).append(idx)
        str_purv = ",".join(ABECEDARY[i] for i in purv) if purv else VOID_STR
        str_mech = ",".join(LOWER_ABECEDARY[i] for i in mech) if mech else VOID_STR
        width = max(len(str_purv), len(str_mech)) + 2
        purv_rows.append(f"|{str_purv:^{width}}|")
        mech_rows.append(f"|{str_mech:^{width}}|")
    return "".join(purv_rows) + "\n" + "".join(mech_rows)




def tensor_product_k(distribuciones: list[np.ndarray]) -> np.ndarray:
    if not distribuciones:
        return np.array([])
    result = distribuciones[0]
    for d in distribuciones[1:]:
        result = np.outer(result, d).flatten()
    return result




class KQNodes(SIA):
    """
    KQNodes: Estrategia híbrida para la k-partición de mínima información (k-MIP).


    Para k=2 delega en QNodes (Queyranne exacto, resultados idénticos).
    Para k>2 selecciona automáticamente entre dos modos según el tamaño
    del sistema:


      Modo exhaustivo (óptimo global):
        Se activa cuando S(2n, k) ≤ 1e5, donde n es el número de nodos
        del subsistema. Genera y evalúa todas las k-particiones posibles
        mediante restricted growth strings, garantizando la k-MIP global.
        Adecuado para validación experimental en sistemas pequeños (típicamente
        n ≤ 5 para cualquier k, n ≤ 6 para k=3).


      Modo greedy (alta calidad, escalable):
        Para sistemas más grandes donde S(2n, k) excede el umbral.
        Parte de la bipartición óptima de Queyranne y subdivide
        iterativamente el grupo cuya partición interna más reduce la
        pérdida EMD. No garantiza optimalidad global pero mantiene
        tiempos de ejecución razonables.


    La evaluación de pérdida es idéntica en ambos modos:
    EMD efecto entre la distribución del subsistema y el vector de
    probabilidades marginales por nodo de la k-partición (equivalente al
    producto tensorial de k distribuciones marginales bajo independencia
    condicional).


    Args:
        tpm: Matriz de probabilidad de transición.
        k: Número de grupos (2 ≤ k ≤ 5).
        busqueda_exhaustiva: Si True fuerza modo exhaustivo;
                            si False fuerza modo greedy;
                            si None (default) selección automática por S(2n,k).
    """


    def __init__(self, tpm: np.ndarray, k: int = 2,
                 busqueda_exhaustiva: bool | None = None):
        super().__init__(tpm)
        self.k = k
        self.busqueda_exhaustiva = busqueda_exhaustiva
        gestor_perfilado.start_session(
            f"{NET_LABEL}{len(tpm[COLS_IDX])}{aplicacion.pagina_red_muestra}"
        )
        self.vertices: set[tuple] = set()
        self.memoria_grupo_candidato: dict = {}
        self.memoria_delta: dict = {}
        self.clave_submodular: list[list[int]] = [[], []]
        self.indices_alcance: np.ndarray
        self.indices_mecanismo: np.ndarray
        self.logger = SafeLogger(KQNODES_STRATEGY_TAG)


    @profile(context={TYPE_TAG: KQNODES_ANALYSIS_TAG})
    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
    ):
        if self.k == 2:
            return self._delegar_qnodes(estado_inicial, condicion, alcance, mecanismo)


        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)


        futuro = tuple(
            (EFFECT, i) for i in self.sia_subsistema.indices_ncubos
        )
        presente = tuple(
            (ACTUAL, i) for i in self.sia_subsistema.dims_ncubos
        )
        vertices = list(presente + futuro)
        self.vertices = set(vertices)


        groups, perdida, dist_part = self.find_k_mip(vertices, self.k)


        return Solution(
            estrategia=KQNODES_LABEL,
            perdida=perdida,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=dist_part,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_kparte_q(groups),
        )


    def _delegar_qnodes(self, estado_inicial: str, condicion: str,
                        alcance: str, mecanismo: str):
        from src.strategies.q_nodes import QNodes
        qn = QNodes(self.tpm)
        sol = qn.aplicar_estrategia(estado_inicial, condicion, alcance, mecanismo)
        sol.estrategia = KQNODES_LABEL
        return sol


    def _usar_exhaustivo(self, num_vertices: int) -> bool:
        """Determina si usar búsqueda exhaustiva según umbral S(2n,k)."""
        if self.busqueda_exhaustiva is True:
            return True
        if self.busqueda_exhaustiva is False:
            return False
        return stirling2(num_vertices, self.k) <= 100_000


    def find_k_mip(
        self, vertices: list, k: int
    ) -> tuple[list[list[tuple[int, int]]], float, np.ndarray]:
        n_vertices = len(vertices)


        if self._usar_exhaustivo(n_vertices):
            return self._exhaustivo(vertices, k)


        return self._greedy(vertices, k)


    def _exhaustivo(
        self, vertices: list, k: int
    ) -> tuple[list[list[tuple[int, int]]], float, np.ndarray]:
        """Modo exhaustivo: genera y evalúa todas las k-particiones posibles.


        Utiliza restricted growth strings para enumerar exactamente S(m,k)
        particiones, donde m = |vertices|. Garantiza la k-MIP óptima global.
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


        return best_groups, best_emd, best_dist


    def _greedy(
        self, vertices: list, k: int
    ) -> tuple[list[list[tuple[int, int]]], float, np.ndarray]:
        """Modo greedy: biparticiones sucesivas desde Queyranne.


        Parte de la bipartición óptima y subdivide iterativamente el
        grupo que más reduce la pérdida EMD. Escalable pero no garantiza
        optimalidad global para k>2.
        """
        mip = self._queyranne(vertices)
        groups = [list(mip), list(set(vertices) - set(mip))]
        best_emd, best_dist = self.evaluate_k_partition(groups)


        while len(groups) < k:
            best_idx = -1
            best_split = None
            best_candidate_emd = float("inf")
            best_candidate_dist = None


            for idx, group in enumerate(groups):
                if len(group) < 2:
                    continue
                candidates = self._enumerate_splits(group)
                for g1, g2 in candidates:
                    candidate = groups[:idx] + groups[idx + 1:] + [g1, g2]
                    emd_val, dist_val = self.evaluate_k_partition(candidate)
                    if emd_val < best_candidate_emd:
                        best_candidate_emd = emd_val
                        best_candidate_dist = dist_val
                        best_idx = idx
                        best_split = (g1, g2)


            if best_idx == -1:
                break


            groups = (
                groups[:best_idx]
                + groups[best_idx + 1:]
                + [list(best_split[0]), list(best_split[1])]
            )
            best_emd = best_candidate_emd
            best_dist = best_candidate_dist


        return groups, best_emd, best_dist


    def _enumerate_splits(
        self, group: list[tuple[int, int]]
    ) -> list[tuple[list[tuple[int, int]], list[tuple[int, int]]]]:
        if len(group) < 2:
            return []
        if len(group) == 2:
            return [([group[0]], [group[1]])]
        mip = self._queyranne(group)
        g1 = list(mip)
        g2 = list(set(group) - set(mip))
        return [(g1, g2)]


    def _queyranne(self, vertices: list[tuple[int, int]]) -> tuple:
        memoria: dict = {}


        for i in range(len(vertices) - 1):
            omegas_ciclo = [vertices[0]]
            deltas_ciclo = vertices[1:]


            emd_particion_candidata = INFTY_POS
            dist_particion_candidata = None


            for j in range(len(deltas_ciclo) - 1):
                emd_local = 1e5
                indice_mip: int = 0


                for k_delta in range(len(deltas_ciclo)):
                    emd_union, emd_delta, dist_marginal_delta = (
                        self._funcion_submodular(deltas_ciclo[k_delta], omegas_ciclo)
                    )
                    emd_iteracion = emd_union - emd_delta


                    if emd_iteracion < emd_local:
                        emd_local = emd_iteracion
                        indice_mip = k_delta


                    emd_particion_candidata = emd_delta
                    dist_particion_candidata = dist_marginal_delta


                omegas_ciclo.append(deltas_ciclo[indice_mip])
                deltas_ciclo.pop(indice_mip)


            memoria[
                tuple(
                    deltas_ciclo[LAST_IDX]
                    if isinstance(deltas_ciclo[LAST_IDX], list)
                    else deltas_ciclo
                )
            ] = emd_particion_candidata, dist_particion_candidata


            par_candidato = (
                [omegas_ciclo[LAST_IDX]]
                if isinstance(omegas_ciclo[LAST_IDX], tuple)
                else omegas_ciclo[LAST_IDX]
            ) + (
                deltas_ciclo[LAST_IDX]
                if isinstance(deltas_ciclo[LAST_IDX], list)
                else deltas_ciclo
            )


            omegas_ciclo.pop()
            omegas_ciclo.append(par_candidato)
            vertices = omegas_ciclo


        return min(memoria, key=lambda k: memoria[k][0])


    def _funcion_submodular(
        self,
        deltas: Union[tuple, list[tuple]],
        omegas: list[Union[tuple, list[tuple]]],
    ):
        self.clave_submodular = [[], []]


        self._definir_clave(deltas)
        idxs_alcance_delta = self.clave_submodular[EFFECT]
        dims_mecanismo_delta = self.clave_submodular[ACTUAL]


        particion_delta = self.sia_subsistema.bipartir(
            np.array(idxs_alcance_delta, dtype=np.int8),
            np.array(dims_mecanismo_delta, dtype=np.int8),
        )
        vector_delta_marginal = particion_delta.distribucion_marginal()
        emd_delta = emd_efecto(vector_delta_marginal, self.sia_dists_marginales)


        for omega in omegas:
            self._definir_clave(omega)


        idxs_alcance_union = self.clave_submodular[EFFECT]
        dims_mecanismo_union = self.clave_submodular[ACTUAL]


        particion_union = self.sia_subsistema.bipartir(
            np.array(idxs_alcance_union, dtype=np.int8),
            np.array(dims_mecanismo_union, dtype=np.int8),
        )
        vector_union_marginal = particion_union.distribucion_marginal()
        emd_union = emd_efecto(vector_union_marginal, self.sia_dists_marginales)


        return emd_union, emd_delta, vector_delta_marginal


    def _definir_clave(self, conjunto: Union[tuple[int, int], list[tuple[int, int]]]):
        if isinstance(conjunto, tuple):
            tiempo, indice = conjunto
            self.clave_submodular[tiempo].append(indice)
        else:
            for tiempo, indice in conjunto:
                self.clave_submodular[tiempo].append(indice)
        self.clave_submodular[ACTUAL].sort()
        self.clave_submodular[EFFECT].sort()
        return self.clave_submodular


    def evaluate_k_partition(
        self, groups: list[list[tuple[int, int]]]
    ) -> tuple[float, np.ndarray]:
        future_to_mech: dict[int, list[int]] = {}
        for group in groups:
            mech = sorted([idx for t, idx in group if t == ACTUAL])
            purv = sorted([idx for t, idx in group if t == EFFECT])
            for p in purv:
                future_to_mech[p] = mech


        n = len(self.sia_subsistema.ncubos)
        part_dist = np.empty(n, dtype=np.float32)


        for i, cube in enumerate(self.sia_subsistema.ncubos):
            if cube.indice in future_to_mech:
                mech = future_to_mech[cube.indice]
                if mech:
                    keep = np.array(mech, dtype=np.int8)
                    marg = np.setdiff1d(cube.dims, keep)
                    marginalized = cube.marginalizar(marg)
                else:
                    marginalized = cube.marginalizar(cube.dims)


                prob = marginalized.data
                if marginalized.dims.size:
                    sub_state = tuple(
                        int(self.sia_subsistema.estado_inicial[j])
                        for j in marginalized.dims
                    )
                    if aplicacion.notacion_indexado == Notation.LIL_ENDIAN.value:
                        sub_state = sub_state[::-1]
                    prob = marginalized.data[sub_state]
                part_dist[i] = 1 - float(prob)
            else:
                part_dist[i] = 0.0


        emd = emd_efecto(part_dist, self.sia_dists_marginales)
        return emd, part_dist


    def nodes_complement(self, nodes: list[tuple[int, int]]) -> list[tuple[int, int]]:
        return list(set(self.vertices) - set(nodes))
