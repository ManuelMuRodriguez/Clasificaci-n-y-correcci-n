"""
comparar_tipos_ventilacion.py
=============================
Compara las tres variables de posición de ventana disponibles en OPC UA:
  - _POS        : posición comandada (escritura al actuador)
  - _POS_VALOR  : posición leída por el sistema (realimentación software)
  - _Encoder    : posición real medida por encoder físico

Disponibilidad de encoders en el dataset:
  - OPC_INVER_UVLAT1N_Encoder   → desde 2024-11-16
  - Resto de encoders            → desde 2025-05-07

Uso
---
  # Un mes concreto — guarda PNG
  python src/comparar_tipos_ventilacion.py --mes 2025-06

  # Todos los meses con encoders disponibles — guarda PNGs + resumen CSV
  python src/comparar_tipos_ventilacion.py --loop

  # Carpeta de salida personalizada
  python src/comparar_tipos_ventilacion.py --loop --output-dir data/tipos_ventilacion/

Conclusión del análisis (2024-12 → 2025-12)
--------------------------------------------
El análisis de MAE respecto al encoder físico revela que ninguna de las dos
señales software es fiable como indicador de posición real:

  - _POS_VALOR (realimentación software): MAE típico del 40–85 % en la mayoría
    de ventanas. Solo UVLAT1N muestra coherencia sostenida (< 2 %), probablemente
    por ser el encoder más antiguo y mejor calibrado.

  - _POS (comando): MAE del 80–99 % de forma sistemática. El actuador no ejecuta
    fielmente la orden recibida, o la señal no refleja la posición física.

Implicación para el modelo de detección de anomalías
-----------------------------------------------------
Las señales de posición de ventana (_POS, _POS_VALOR) se excluyen del modelo
durante el período de estudio (2023-2024), ya que los encoders físicos no
estaban disponibles y las señales software demuestran desviaciones no asumibles.
El efecto de la ventilación queda recogido implícitamente en las variables
climáticas interiores (XTINV, XHINV, XCO2I), que actúan como proxies del
estado real de apertura. Esta limitación se documenta explícitamente en el paper
y abre la puerta a incorporar la posición real del encoder en trabajo futuro.
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_single_opcua_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OPCUA_DIR = PROJECT_ROOT / "Dataset" / "OPCUA"

# Mapeo: nombre_corto → (col_POS, col_POS_VALOR, col_Encoder)
# Encoder = None si aún no existe para esa ventana
VENTANAS = {
    # Centrales grupo 1
    "UVCEN1_1": ("OPC_UVCEN1_1_POS",  "OPC_UVCEN1_1_POS_VALOR",  "OPC_INVER_UVCEN11_Encoder"),
    "UVCEN1_2": ("OPC_UVCEN1_2_POS",  "OPC_UVCEN1_2_POS_VALOR",  "OPC_INVER_UVCEN12_Encoder"),
    "UVCEN1_3": ("OPC_UVCEN1_3_POS",  "OPC_UVCEN1_3_POS_VALOR",  "OPC_INVER_UVCEN13_Encoder"),
    # Centrales grupo 2
    "UVCEN2_1": ("OPC_UVCEN2_1_POS",  "OPC_UVCEN2_1_POS_VALOR",  "OPC_INVER_UVCEN21_Encoder"),
    "UVCEN2_2": ("OPC_UVCEN2_2_POS",  "OPC_UVCEN2_2_POS_VALOR",  "OPC_INVER_UVCEN22_Encoder"),
    "UVCEN2_3": ("OPC_UVCEN2_3_POS",  "OPC_UVCEN2_3_POS_VALOR",  "OPC_INVER_UVCEN23_Encoder"),
    # Laterales norte
    "UVLAT1N":  ("OPC_UVLAT1N_POS",   "OPC_UVLAT1N_POS_VALOR",   "OPC_INVER_UVLAT1N_Encoder"),
    "UVLAT1ON": ("OPC_UVLAT1ON_POS",  "OPC_UVLAT1ON_POS_VALOR",  "OPC_INVER_UVLAT1NO_Encoder"),
    "UVLAT2N":  ("OPC_UVLAT2N_POS",   "OPC_UVLAT2N_POS_VALOR",   "OPC_INVER_UVLAT2N_Encoder"),
    # Laterales sur
    "UVLAT1S":  ("OPC_UVLAT1S_POS",   "OPC_UVLAT1S_POS_VALOR",   "OPC_INVER_UVLAT1S_Encoder"),
    "UVLAT1OS": ("OPC_UVLAT1OS_POS",  "OPC_UVLAT1OS_POS_VALOR",  "OPC_INVER_UVLAT1SO_Encoder"),
    "UVLAT2S":  ("OPC_UVLAT2S_POS",   "OPC_UVLAT2S_POS_VALOR",   "OPC_INVER_UVLAT2S_Encoder"),
    # Lateral este
    "UVLAT2E":  ("OPC_UVLAT2E_POS",   "OPC_UVLAT2E_POS_VALOR",   "OPC_INVER_UVLAT2E_Encoder"),
}

GRUPOS = {
    "Centrales S1": ["UVCEN1_1", "UVCEN1_2", "UVCEN1_3"],
    "Centrales S2": ["UVCEN2_1", "UVCEN2_2", "UVCEN2_3"],
    "Laterales N":  ["UVLAT1N",  "UVLAT1ON", "UVLAT2N"],
    "Laterales S":  ["UVLAT1S",  "UVLAT1OS", "UVLAT2S"],
    "Lateral E":    ["UVLAT2E"],
}

# Encoders disponibles por fecha
ENCODER_INICIO = {
    "OPC_INVER_UVLAT1N_Encoder":  pd.Timestamp("2024-11-16"),
    "OPC_INVER_UVCEN11_Encoder":  pd.Timestamp("2025-05-07"),
    "OPC_INVER_UVCEN12_Encoder":  pd.Timestamp("2025-05-07"),
    "OPC_INVER_UVCEN13_Encoder":  pd.Timestamp("2025-05-07"),
    "OPC_INVER_UVCEN21_Encoder":  pd.Timestamp("2025-05-07"),
    "OPC_INVER_UVCEN22_Encoder":  pd.Timestamp("2025-05-08"),
    "OPC_INVER_UVCEN23_Encoder":  pd.Timestamp("2025-05-08"),
    "OPC_INVER_UVLAT1NO_Encoder": pd.Timestamp("2025-05-07"),
    "OPC_INVER_UVLAT1S_Encoder":  pd.Timestamp("2025-05-07"),
    "OPC_INVER_UVLAT1SO_Encoder": pd.Timestamp("2025-06-02"),
    "OPC_INVER_UVLAT2E_Encoder":  pd.Timestamp("2025-05-07"),
    "OPC_INVER_UVLAT2N_Encoder":  pd.Timestamp("2025-05-07"),
    "OPC_INVER_UVLAT2S_Encoder":  pd.Timestamp("2025-05-08"),
}


def _meses_con_encoders() -> list[str]:
    """Meses desde el primer encoder disponible (nov 2024) hasta el último fichero OPC UA."""
    primer_encoder = min(ENCODER_INICIO.values())
    meses = set()
    for f in sorted(OPCUA_DIR.rglob("OPC_*.txt")):
        nombre = f.stem  # OPC_YYYYMMDD
        if len(nombre) >= 12:
            try:
                fecha = pd.to_datetime(nombre[4:], format="%Y%m%d")
                if fecha >= primer_encoder:
                    meses.add(nombre[4:8] + "-" + nombre[8:10])
            except:
                pass
    return sorted(meses)


def _grupo_tiene_encoder_en_mes(ventanas_grupo: list[str], inicio: pd.Timestamp) -> bool:
    """Devuelve True si al menos un encoder del grupo tiene datos reales en ese mes."""
    for vent in ventanas_grupo:
        _, _, col_enc = VENTANAS[vent]
        if col_enc in ENCODER_INICIO and inicio >= ENCODER_INICIO[col_enc]:
            return True
    return False


def _cargar_mes_opcua(mes: str) -> pd.DataFrame | None:
    """Carga solo los ficheros OPC UA del mes indicado (patrón OPC_YYYYMM*.txt)."""
    mes_str = mes.replace("-", "")
    ficheros = sorted(OPCUA_DIR.rglob(f"OPC_{mes_str}*.txt"))
    if not ficheros:
        return None

    dfs = []
    for f in ficheros:
        try:
            dfs.append(load_single_opcua_file(f))
        except Exception as e:
            logger.warning(f"  Error en {f.name}: {e}")

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True, sort=False)
    df.columns = [c.lstrip("-") for c in df.columns]
    return df.sort_values("Fecha").reset_index(drop=True)


def comparar_mes(mes: str, output: Path) -> list[dict]:
    """Genera y guarda las gráficas del mes. Devuelve lista de dicts con métricas por ventana."""
    inicio = pd.Timestamp(mes + "-01")
    fin    = (inicio + pd.offsets.MonthEnd(0)).replace(hour=23, minute=59, second=59)

    logger.info(f"Cargando OPC UA para {mes}...")
    df_raw = _cargar_mes_opcua(mes)
    if df_raw is None or len(df_raw) == 0:
        logger.warning(f"Sin ficheros OPC UA para {mes} — omitiendo.")
        return []

    df_raw = df_raw.set_index("Fecha").sort_index()

    # Resamplear a 1 min
    cols_numericas = [c for c in df_raw.columns if df_raw[c].dtype != object]
    df = df_raw[cols_numericas].resample("1min").mean()

    registros_mes: list[dict] = []

    # Una figura por grupo
    for nombre_grupo, ventanas_grupo in GRUPOS.items():
        n = len(ventanas_grupo)
        fig, axes = plt.subplots(n, 1, figsize=(16, 4 * n), sharex=True)
        if n == 1:
            axes = [axes]

        # Saltar grupo si ningún encoder tiene datos reales en este mes
        if not _grupo_tiene_encoder_en_mes(ventanas_grupo, inicio):
            plt.close(fig)
            logger.info(f"  Grupo '{nombre_grupo}' sin encoders en {mes} — omitido.")
            continue

        tiene_datos = False

        for ax, vent in zip(axes, ventanas_grupo):
            col_pos, col_valor, col_enc = VENTANAS[vent]

            enc_disponible = (
                col_enc in df.columns and
                col_enc in ENCODER_INICIO and
                inicio >= ENCODER_INICIO[col_enc]
            )

            metricas_ax = []
            fila: dict = {"Mes": mes, "Grupo": nombre_grupo, "Ventana": vent,
                          "MAE_VALOR_vs_Enc": None, "MAE_POS_vs_Enc": None}

            s_pos = None
            s_val = None
            s_enc = None

            if col_pos in df.columns:
                s_pos = pd.to_numeric(df[col_pos], errors="coerce")
                ax.plot(s_pos.index, s_pos.values,
                        color="black", linewidth=0.8, alpha=0.7, label="POS (comando)")
                tiene_datos = True

            if col_valor in df.columns:
                s_val = pd.to_numeric(df[col_valor], errors="coerce")
                ax.plot(s_val.index, s_val.values,
                        color="tab:orange", linewidth=0.9, alpha=0.85, label="POS VALOR (lectura)")
                tiene_datos = True

            if enc_disponible:
                s_enc = pd.to_numeric(df[col_enc], errors="coerce")
                s_enc = s_enc.where(s_enc != 0)
                ax.plot(s_enc.index, s_enc.values,
                        color="tab:blue", linewidth=1.0, alpha=0.9, label="Encoder (real)")

                if s_val is not None:
                    mask = s_val.notna() & s_enc.notna()
                    if mask.sum() > 10:
                        mae_ve = (s_val[mask] - s_enc[mask]).abs().mean()
                        fila["MAE_VALOR_vs_Enc"] = round(mae_ve, 2)
                        metricas_ax.append(f"MAE _VALOR vs _Enc={mae_ve:.2f}%")

                if s_pos is not None:
                    mask2 = s_pos.notna() & s_enc.notna()
                    if mask2.sum() > 10:
                        mae_pe = (s_pos[mask2] - s_enc[mask2]).abs().mean()
                        fila["MAE_POS_vs_Enc"] = round(mae_pe, 2)
                        metricas_ax.append(f"MAE _POS vs _Enc={mae_pe:.2f}%")

            if enc_disponible:
                registros_mes.append(fila)

            titulo_ax = vent
            if metricas_ax:
                titulo_ax += "  |  " + "   ".join(metricas_ax)
            ax.set_title(titulo_ax, fontsize=8, loc="left")
            ax.set_ylabel("Posición (%)")
            ax.set_ylim(-5, 115)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=9, framealpha=0.9,
                      edgecolor="gray", fancybox=True)

        if not tiene_datos:
            plt.close(fig)
            continue

        n_dias = (fin - inicio).days + 1
        locator   = mdates.DayLocator(interval=max(1, n_dias // 15))
        formatter = mdates.DateFormatter("%d/%m")
        axes[-1].xaxis.set_major_locator(locator)
        axes[-1].xaxis.set_major_formatter(formatter)
        axes[-1].set_xlabel("Fecha (UTC)")
        fig.autofmt_xdate(rotation=30)

        fig.suptitle(
            f"Ventanas {nombre_grupo} — {mes}\n"
            f"negro=_POS (comando)  |  naranja=_POS_VALOR (lectura)  |  azul=_Encoder (real)",
            fontsize=10,
        )
        fig.tight_layout()

        nombre_grupo_safe = nombre_grupo.replace(" ", "_").replace("/", "_")
        out = output.parent / f"{output.stem}_{nombre_grupo_safe}{output.suffix}"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=130, bbox_inches="tight")
        logger.info(f"  Guardada: {out}")
        plt.close(fig)

    return registros_mes


def _imprimir_tabla_metricas(registros: list[dict], output_dir: Path) -> None:
    """Imprime y guarda un CSV con el resumen de métricas por ventana y mes."""
    if not registros:
        logger.warning("Sin métricas que mostrar.")
        return

    df = pd.DataFrame(registros)
    df = df.sort_values(["Mes", "Grupo", "Ventana"]).reset_index(drop=True)

    # Solo filas con al menos una métrica
    df_con_enc = df[df["MAE_VALOR_vs_Enc"].notna() | df["MAE_POS_vs_Enc"].notna()].copy()

    print("\n" + "=" * 75)
    print("RESUMEN DE MÉTRICAS (meses con encoder disponible)")
    print("=" * 75)
    with pd.option_context("display.max_rows", None, "display.float_format", "{:.2f}".format):
        print(df_con_enc.to_string(index=False))
    print("=" * 75)

    csv_out = output_dir / "resumen_metricas.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df_con_enc.to_csv(csv_out, index=False)
    logger.info(f"Resumen guardado en: {csv_out}")


def main():
    parser = argparse.ArgumentParser(
        description="Compara _POS, _POS_VALOR y _Encoder de las ventanas OPC UA"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mes",  help="Mes concreto en formato YYYY-MM")
    group.add_argument("--loop", action="store_true",
                       help="Todos los meses desde que hay encoders disponibles")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "tipos_ventilacion",
    )
    args = parser.parse_args()

    if not OPCUA_DIR.exists():
        logger.error(f"Directorio OPC UA no encontrado: {OPCUA_DIR}")
        sys.exit(1)

    todos_registros: list[dict] = []

    if args.mes:
        output = args.output_dir / f"tipos_{args.mes}.png"
        registros = comparar_mes(args.mes, output)
        todos_registros.extend(registros)
    else:
        meses = _meses_con_encoders()
        logger.info(f"Meses con encoders disponibles: {meses}")
        for mes in meses:
            output = args.output_dir / f"tipos_{mes}.png"
            registros = comparar_mes(mes, output)
            todos_registros.extend(registros)
        logger.info(f"\nFin. Gráficas en {args.output_dir}/")

    _imprimir_tabla_metricas(todos_registros, args.output_dir)


if __name__ == "__main__":
    main()
