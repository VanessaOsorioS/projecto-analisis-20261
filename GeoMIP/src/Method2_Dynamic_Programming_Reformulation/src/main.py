from src.models.base.application import aplicacion
from src.controllers.manager import Manager
from src.controllers.strategies.geometric import GeometricSIA
from src.controllers.strategies.q_nodes import QNodes
# Optional import: this project often runs only geometric strategy.
try:
    from src.controllers.strategies.phi import Phi
except Exception:
    Phi = None
import multiprocessing
import numpy as np
import pandas as pd
import os
import re
from pathlib import Path


METHOD2_ROOT = Path(__file__).resolve().parents[1]
GEOMIP_ROOT = Path(__file__).resolve().parents[3]

def convertir_a_binario(texto, n_bits=20):
    posiciones = "ABCDEFGHIJKLMNOPQRST"[:n_bits]
    binario = ["0"] * n_bits
    for letra in texto:
        if letra in posiciones:
            binario[posiciones.index(letra)] = "1"
    return "".join(binario)

def ejecutar_con_tiempo(config_sistema, condiciones, alcance, mecanismo, resultado_queue, tpm, k=None):
    try:
        analizador_fi = GeometricSIA(config_sistema)
        if k is not None:
            analizador_fi.k_particiones = k
        sia_dos = analizador_fi.aplicar_estrategia(condiciones, alcance, mecanismo, tpm)
        resultado_queue.put({
            "particion": sia_dos.particion,
            "perdida": str(sia_dos.perdida).replace('.', ','),
            "tiempo": str(sia_dos.tiempo_ejecucion).replace('.', ','),
        })

    except Exception as e:
        resultado_queue.put({
            "particion": None,
            "perdida": None,
            "tiempo": None,
        })

def resolver_tpm_path(estado_inicio: str, letra: str | None = None) -> Path:
    pagina = letra or aplicacion.pagina_sample_network
    sample_name = f"N{len(estado_inicio)}{pagina}.csv"
    candidates = (
        METHOD2_ROOT / "src" / ".samples" / sample_name,
        METHOD2_ROOT / ".samples" / sample_name,
        GEOMIP_ROOT / "data" / "samples" / sample_name,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No se encontró la TPM '{sample_name}'. Busqué en: {', '.join(str(c) for c in candidates)}"
    )


def inferir_estado_inicial() -> str:
    """Infer an initial state from available datasets (prefers largest NxA.csv)."""
    sample_dirs = (
        METHOD2_ROOT / "src" / ".samples",
        METHOD2_ROOT / ".samples",
        GEOMIP_ROOT / "data" / "samples",
    )
    pattern = re.compile(r"N(\d+)[A-Z]\.csv$")
    available_sizes = []

    for sample_dir in sample_dirs:
        if not sample_dir.exists():
            continue
        for sample_file in sample_dir.glob("N*.csv"):
            match = pattern.match(sample_file.name)
            if match:
                available_sizes.append(int(match.group(1)))

    if not available_sizes:
        raise FileNotFoundError("No hay archivos de muestras TPM disponibles en data/samples ni .samples.")

    n_bits = max(available_sizes)
    return "1" + ("0" * (n_bits - 1))


def ejecutar_desde_excel(
    ruta_excel: Path,
    ruta_salida: Path,
    inicio=0,
    cantidad=50,
    estado_inicio: str | None = None,
    condiciones: str | None = None,
):
    df = pd.read_excel(ruta_excel, sheet_name=8, usecols="B", skiprows=3, names=["Subsistema"]) #! here
    filas = df["Subsistema"].dropna().tolist()
    filas = filas[inicio:inicio + cantidad]
    resultados = []

    estado_inicio = estado_inicio or inferir_estado_inicial()
    condiciones = condiciones or ("1" * len(estado_inicio))
    tpm_path = resolver_tpm_path(estado_inicio)
    tpm = np.genfromtxt(tpm_path, delimiter=",")

    for i, fila in enumerate(filas, start=inicio + 1):
        partes = fila.split("|")
        if len(partes) != 2:
            continue

        alcance = convertir_a_binario(partes[0][:len(partes[0]) - 3], n_bits=len(estado_inicio))
        mecanismo = convertir_a_binario(partes[1][:len(partes[1]) - 1], n_bits=len(estado_inicio))
        print(f"Iteración {i} - Alcance: {alcance}, Mecanismo: {mecanismo}")

        config_sistema = Manager(estado_inicial=estado_inicio)

        resultado_queue = multiprocessing.Queue()
        proceso = multiprocessing.Process(target=ejecutar_con_tiempo, args=(config_sistema, condiciones, alcance, mecanismo, resultado_queue, tpm))
        
        proceso.start()
        proceso.join(timeout=3600)  

        if proceso.is_alive():
            print(f"Iteración {i} - Tiempo límite alcanzado, terminando proceso...")
            proceso.terminate()
            proceso.join()
            resultado = {"perdida": None, "tiempo": None, "particion": None}
        else:
            resultado = (
                resultado_queue.get()
                if not resultado_queue.empty()
                else {"perdida": None, "tiempo": None, "particion": None}
            )

        resultados.append({
            "Iteración": i,
            "Alcance": alcance,
            "Mecanismo": mecanismo,
            "Partición": resultado["particion"],
            "Pérdida": resultado["perdida"],
            "Tiempo de ejecución (s)": resultado["tiempo"],
        })
    df_resultados = pd.DataFrame(resultados)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    df_resultados.to_excel(ruta_salida, index=False)
    print(f"Resultados guardados en {ruta_salida}")

def iniciar(
    estado_inicio: str | None = None,
    condiciones: str | None = None,
):
    estado_inicio = estado_inicio or os.getenv("GEOMIP_ESTADO")

    ruta_entrada = Path(
        os.getenv(
            "GEOMIP_INPUT_XLSX",
            str(GEOMIP_ROOT / "results" / "Pruebas_Metodo2.xlsx"),
        )
    )
    ruta_salida = Path(
        os.getenv(
            "GEOMIP_OUTPUT_XLSX",
            str(GEOMIP_ROOT / "results" / "resultados_Geometric.xlsx"),
        )
    )
    ejecutar_desde_excel(
        ruta_entrada,
        ruta_salida,
        estado_inicio=estado_inicio,
        condiciones=condiciones,
    )