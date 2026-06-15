import sys
import os
from pathlib import Path
import multiprocessing
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from src.controllers.manager import Manager
from src.controllers.strategies.k_geometric import KGeoMIP
from src.main import (
    convertir_a_binario,
    resolver_tpm_path,
    inferir_estado_inicial,
    GEOMIP_ROOT,
)
from src.models.base.application import aplicacion


def ejecutar_con_tiempo(
    config_sistema, condiciones, alcance, mecanismo, resultado_queue, tpm, k
):
    try:
        analizador = KGeoMIP(config_sistema, k=k)
        sol = analizador.aplicar_estrategia(condiciones, alcance, mecanismo, tpm)
        resultado_queue.put(
            {
                "particion": sol.particion,
                "perdida": str(sol.perdida).replace(".", ","),
                "tiempo": str(sol.tiempo_ejecucion).replace(".", ","),
            }
        )
    except Exception as e:
        resultado_queue.put({"particion": None, "perdida": None, "tiempo": None})


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    iteracion_especifica = int(sys.argv[2]) if len(sys.argv) > 2 else None
    ruta_excel = Path(
        os.getenv(
            "GEOMIP_INPUT_XLSX",
            str(GEOMIP_ROOT / "results" / "Pruebas_Metodo2.xlsx"),
        )
    )
    sufijo = f"_it{iteracion_especifica}" if iteracion_especifica else ""
    ruta_salida = GEOMIP_ROOT / "results" / f"resultados_KGeoMIP_k{k}{sufijo}.xlsx"

    estado_inicio = inferir_estado_inicial()
    condiciones = "1" * len(estado_inicio)
    tpm_path = resolver_tpm_path(estado_inicio)
    tpm = np.genfromtxt(tpm_path, delimiter=",")

    df = pd.read_excel(ruta_excel, sheet_name=8, usecols="B", skiprows=3, names=["Subsistema"])
    filas = df["Subsistema"].dropna().tolist()
    resultados = []

    for i, fila in enumerate(filas, start=1):
        if iteracion_especifica is not None and i != iteracion_especifica:
            continue
        partes = fila.split("|")
        if len(partes) != 2:
            continue

        alcance = convertir_a_binario(
            partes[0][: len(partes[0]) - 3], n_bits=len(estado_inicio)
        )
        mecanismo = convertir_a_binario(
            partes[1][: len(partes[1]) - 1], n_bits=len(estado_inicio)
        )
        print(f"Iteración {i} - k={k} - Alcance: {alcance}, Mecanismo: {mecanismo}")

        config_sistema = Manager(estado_inicial=estado_inicio)
        resultado_queue = multiprocessing.Queue()
        proceso = multiprocessing.Process(
            target=ejecutar_con_tiempo,
            args=(config_sistema, condiciones, alcance, mecanismo, resultado_queue, tpm, k),
        )
        proceso.start()
        proceso.join(timeout=3600)

        if proceso.is_alive():
            print(f"Iteración {i} - Tiempo límite alcanzado, terminando...")
            proceso.terminate()
            proceso.join()
            resultado = {"perdida": None, "tiempo": None, "particion": None}
        else:
            resultado = (
                resultado_queue.get()
                if not resultado_queue.empty()
                else {"perdida": None, "tiempo": None, "particion": None}
            )

        resultados.append(
            {
                "Iteración": i,
                "Alcance": alcance,
                "Mecanismo": mecanismo,
                "Partición": resultado["particion"],
                "Pérdida": resultado["perdida"],
                "Tiempo de ejecución (s)": resultado["tiempo"],
            }
        )

    df_resultados = pd.DataFrame(resultados)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    df_resultados.to_excel(ruta_salida, index=False)
    print(f"\nResultados guardados en {ruta_salida}")


if __name__ == "__main__":
    main()
