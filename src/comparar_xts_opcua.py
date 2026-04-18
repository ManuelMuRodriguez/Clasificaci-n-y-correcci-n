"""
comparar_xts_opcua.py
=====================
Compara TODAS las columnas TEMP_SUELO presentes en los ficheros raw de SCADA
y de OPC UA para determinar qué columna OPC UA corresponde a cada columna SCADA.

Uso
---
  # Un mes concreto
  python src/comparar_xts_opcua.py --mes 2024-10

  # Todos los meses con OPC UA disponible (loop automático)
  python src/comparar_xts_opcua.py --loop

Salida
------
  - Tabla con MAE y correlación de cada columna OPC UA vs SCADA S1
  - Gráfica por mes guardada en data/suelo_meses/comparacion_YYYY-MM.png
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

from src.data_loader import load_all_files, load_all_opcua_files

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _filtrar_mes(df: pd.DataFrame, col_fecha: str, mes: str) -> pd.DataFrame:
    inicio = pd.Timestamp(mes + "-01")
    fin    = (inicio + pd.offsets.MonthEnd(0)).replace(hour=23, minute=59, second=59)
    return df[(df[col_fecha] >= inicio) & (df[col_fecha] <= fin)].copy()


def _meses_disponibles_opcua(opcua_dir: Path) -> list[str]:
    """Devuelve lista de meses 'YYYY-MM' para los que hay ficheros OPC UA."""
    meses = set()
    for f in sorted(opcua_dir.rglob("OPC_*.txt")):
        # Nombre: OPC_YYYYMMDD.txt → extraer YYYY-MM
        nombre = f.stem  # OPC_20241015
        if len(nombre) >= 11:
            meses.add(nombre[4:8] + "-" + nombre[8:10])
    return sorted(meses)


def comparar_mes(mes: str, scada_dir: Path, opcua_dir: Path, output: Path, show: bool = True):
    mes_str = mes.replace("-", "")

    # ── SCADA raw ─────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"MES: {mes}")
    logger.info(f"{'='*60}")
    logger.info(f"Cargando SCADA (patrón AGROCONNECT_{mes_str}*.xlsx)...")
    try:
        df_scada_raw = load_all_files(
            dataset_dir=scada_dir,
            pattern=f"AGROCONNECT_{mes_str}*.xlsx",
            show_progress=False,
        )
    except FileNotFoundError:
        logger.warning(f"  Sin ficheros SCADA para {mes} — omitiendo.")
        return

    df_scada_raw = df_scada_raw.rename(columns={"FECHA": "Fecha"})
    df_scada_raw = _filtrar_mes(df_scada_raw, "Fecha", mes)
    cols_suelo_scada = [
        c for c in df_scada_raw.columns
        if "TEMP_SUELO" in c.upper() and "SUELO30" not in c.upper()
    ]
    if not cols_suelo_scada:
        logger.warning(f"  Sin columnas TEMP_SUELO en SCADA para {mes}.")
        return
    logger.info(f"  SCADA TEMP_SUELO: {cols_suelo_scada}")

    # ── OPC UA raw ────────────────────────────────────────────────────────────
    logger.info(f"Cargando OPC UA...")
    try:
        df_opc_raw = load_all_opcua_files(opcua_dir=opcua_dir, show_progress=False)
    except FileNotFoundError:
        logger.warning(f"  Sin ficheros OPC UA — omitiendo.")
        return

    df_opc_raw = _filtrar_mes(df_opc_raw, "Fecha", mes)
    cols_suelo_opc = [
        c for c in df_opc_raw.columns
        if "TEMP_SUELO" in c.upper() and "SUELO30" not in c.upper()
    ]
    if not cols_suelo_opc:
        logger.warning(f"  Sin columnas TEMP_SUELO en OPC UA para {mes}.")
        return
    logger.info(f"  OPC UA TEMP_SUELO: {cols_suelo_opc}")

    # ── Resamplear y alinear ──────────────────────────────────────────────────
    df_s = df_scada_raw[["Fecha"] + cols_suelo_scada].set_index("Fecha").resample("1min").mean()
    df_o = df_opc_raw[["Fecha"]  + cols_suelo_opc ].set_index("Fecha").resample("1min").mean()

    idx_comun = df_s.index.intersection(df_o.index)
    if len(idx_comun) < 100:
        logger.warning(f"  Menos de 100 timestamps comunes para {mes} — omitiendo.")
        return

    df_s = df_s.loc[idx_comun]
    df_o = df_o.loc[idx_comun]

    # ── Métricas ──────────────────────────────────────────────────────────────
    ref_col = "INVER_TEMP_SUELO5_S1" if "INVER_TEMP_SUELO5_S1" in df_s.columns else cols_suelo_scada[0]
    xts_ref = df_s[ref_col]

    resultados = []
    for col in cols_suelo_opc:
        serie = df_o[col]
        mask  = xts_ref.notna() & serie.notna()
        n     = mask.sum()
        if n < 100:
            resultados.append({"OPC UA": col, "MAE (°C)": float("nan"), "corr": float("nan")})
            continue
        mae  = (xts_ref[mask] - serie[mask]).abs().mean()
        corr = xts_ref[mask].corr(serie[mask])
        resultados.append({"OPC UA": col, "MAE (°C)": round(mae, 4), "corr": round(corr, 4)})

    df_res = pd.DataFrame(resultados).sort_values("MAE (°C)")
    print(f"\n--- {mes} ---")
    print(df_res.to_string(index=False))

    # ── Gráfica ───────────────────────────────────────────────────────────────
    cmap_opc   = plt.get_cmap("tab10")
    gray_shades = ["black", "#555555", "#999999", "#bbbbbb"]

    fig, ax = plt.subplots(figsize=(16, 5))

    for i, col in enumerate(cols_suelo_scada):
        lw = 2.0 if i == 0 else 1.2
        ax.plot(
            df_s.index.to_numpy(), df_s[col].to_numpy(),
            color=gray_shades[i % len(gray_shades)], linewidth=lw, zorder=10 - i,
            label=f"SCADA  {col}",
        )

    for i, col in enumerate(cols_suelo_opc):
        fila = df_res[df_res["OPC UA"] == col]
        mae_str  = f"{fila['MAE (°C)'].values[0]:.4f}" if not fila.empty and pd.notna(fila["MAE (°C)"].values[0]) else "n/d"
        corr_str = f"{fila['corr'].values[0]:.4f}" if not fila.empty and pd.notna(fila["corr"].values[0]) else "n/d"
        ax.plot(
            df_o.index.to_numpy(), df_o[col].to_numpy(),
            color=cmap_opc(i), linewidth=0.8, alpha=0.85,
            label=f"{col}  |  MAE={mae_str} °C  corr={corr_str}",
        )

    ax.set_ylabel("Temperatura suelo (°C)")
    ax.set_xlabel("Fecha (UTC)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9, edgecolor="gray", fancybox=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    fig.autofmt_xdate(rotation=30)
    ax.set_title(
        f"TEMP_SUELO — SCADA (gris/negro) vs OPC UA (colores) — {mes}\n"
        f"Línea negra = referencia SCADA S1.",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.15, 1, 1])

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130, bbox_inches="tight")
    logger.info(f"  Guardada: {output}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compara TEMP_SUELO SCADA vs OPC UA")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mes",  help="Mes concreto en formato YYYY-MM")
    group.add_argument("--loop", action="store_true", help="Todos los meses con OPC UA disponible")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "suelo_meses",
        help="Carpeta donde guardar las gráficas en modo --loop",
    )
    parser.add_argument(
        "--scada-dir",
        type=Path,
        default=PROJECT_ROOT / "Dataset" / "SCADA",
    )
    parser.add_argument(
        "--opcua-dir",
        type=Path,
        default=PROJECT_ROOT / "Dataset" / "OPCUA",
    )
    args = parser.parse_args()

    if args.mes:
        output = PROJECT_ROOT / "data" / "comparacion_xts_opcua.png"
        comparar_mes(args.mes, args.scada_dir, args.opcua_dir, output, show=True)
    else:
        meses = _meses_disponibles_opcua(args.opcua_dir)
        if not meses:
            logger.error(f"No se encontraron ficheros OPC UA en {args.opcua_dir}")
            sys.exit(1)
        logger.info(f"Meses con OPC UA: {meses}")
        for mes in meses:
            output = args.output_dir / f"comparacion_{mes}.png"
            comparar_mes(mes, args.scada_dir, args.opcua_dir, output, show=False)
        logger.info(f"\nFin. Gráficas guardadas en {args.output_dir}/")


if __name__ == "__main__":
    main()
