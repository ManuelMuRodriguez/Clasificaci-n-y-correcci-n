"""
comparar_ventanas_opcua.py
==========================
Compara UVENT_cen y UVENT_lN entre SCADA y OPC UA mes a mes para detectar
períodos donde OPC UA tiene datos incorrectos (ceros, offset, etc.).

Usa los CSV ya procesados (no los raw) porque UVENT es una media de múltiples
columnas _POS que ya está calculada en el pipeline.

Uso
---
  # Un mes concreto
  python src/comparar_ventanas_opcua.py --mes 2024-10

  # Todos los meses en bucle (guarda PNGs en data/ventanas_meses/)
  python src/comparar_ventanas_opcua.py --loop

  # Solo períodos de ensayo de ventilación (lee el Excel de ensayos)
  python src/comparar_ventanas_opcua.py --ensayos
  python src/comparar_ventanas_opcua.py --ensayos --output-dir data/ensayos_ventanas/
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

VARIABLES_VENTANA = ["UVENT_cen", "UVENT_lN"]
CSV_SCADA    = PROJECT_ROOT / "data" / "scada_2024_03_06-2025_11_30_1min.csv"
CSV_OPCUA    = PROJECT_ROOT / "data" / "opcua_2024_03_06-2025_11_30_1min.csv"
ENSAYOS_XLSX = PROJECT_ROOT / "Dataset" / "Ensayos" / "Lista de ensayos Invernadero AgroConnect.xlsx"


def _cargar_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["Fecha"] + VARIABLES_VENTANA)
    df["Fecha"] = pd.to_datetime(df["Fecha"], dayfirst=True, errors="coerce")
    return df.dropna(subset=["Fecha"]).set_index("Fecha").sort_index()


def _meses_disponibles(df_opc: pd.DataFrame) -> list[str]:
    return sorted(df_opc.index.to_period("M").unique().astype(str).tolist())


def _cargar_ensayos_ventilacion() -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Lee el Excel de ensayos y devuelve los períodos de ventilación agrupados."""
    if not ENSAYOS_XLSX.exists():
        logger.error(f"Excel de ensayos no encontrado: {ENSAYOS_XLSX}")
        sys.exit(1)

    df = pd.read_excel(ENSAYOS_XLSX, header=None).dropna(subset=[1])
    df = df[df[1] != "Fecha"].copy()
    df.columns = ["_", "fecha", "tipo"]
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha"])

    # Filtrar solo ventilación
    mask_vent = df["tipo"].str.lower().str.contains("ventil", na=False)
    df_vent = df[mask_vent].sort_values("fecha").reset_index(drop=True)

    if df_vent.empty:
        logger.warning("No se encontraron ensayos de ventilación en el Excel.")
        return []

    # Agrupar días consecutivos en períodos
    periodos = []
    inicio = df_vent.iloc[0]["fecha"]
    fin    = df_vent.iloc[0]["fecha"]
    tipo   = df_vent.iloc[0]["tipo"]

    for _, row in df_vent.iloc[1:].iterrows():
        diff = (row["fecha"] - fin).days
        if diff <= 1:
            fin = row["fecha"]
        else:
            periodos.append((inicio, fin, tipo))
            inicio = row["fecha"]
            fin    = row["fecha"]
            tipo   = row["tipo"]
    periodos.append((inicio, fin, tipo))

    return periodos


def comparar_rango(label: str, inicio: pd.Timestamp, fin: pd.Timestamp,
                   df_s: pd.DataFrame, df_o: pd.DataFrame,
                   output: Path, show: bool = True):
    """Compara un rango de fechas concreto (no necesariamente un mes completo)."""
    fin_inc = fin.replace(hour=23, minute=59, second=59)

    s = df_s.loc[inicio:fin_inc]
    o = df_o.loc[inicio:fin_inc]

    if len(s) == 0 and len(o) == 0:
        logger.warning(f"Sin datos para {label} en ninguna fuente — omitiendo.")
        return

    # Métricas
    idx_comun = s.index.intersection(o.index)
    print(f"\n--- {label} ---")
    for var in VARIABLES_VENTANA:
        if var not in s.columns or var not in o.columns:
            continue
        ref  = s.loc[idx_comun, var]
        cand = o.loc[idx_comun, var]
        mask = ref.notna() & cand.notna()
        if mask.sum() < 10:
            print(f"  {var}: sin datos suficientes")
            continue
        mae  = (ref[mask] - cand[mask]).abs().mean()
        corr = ref[mask].corr(cand[mask])
        print(f"  {var}: MAE={mae:.2f}  corr={corr:.4f}")

    # Gráfica
    n = len(VARIABLES_VENTANA)
    fig, axes = plt.subplots(n, 1, figsize=(16, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    n_dias = (fin_inc - inicio).days + 1
    locator   = mdates.HourLocator(interval=6) if n_dias <= 3 else mdates.DayLocator(interval=1)
    formatter = mdates.DateFormatter("%d/%m %H:%M") if n_dias <= 3 else mdates.DateFormatter("%d/%m")

    for ax, var in zip(axes, VARIABLES_VENTANA):
        if var in s.columns:
            ax.plot(s.index.to_numpy(), s[var].to_numpy(),
                    color="black", linewidth=0.9, alpha=0.85, label=f"SCADA {var}")
        if var in o.columns:
            ax.plot(o.index.to_numpy(), o[var].to_numpy(),
                    color="tab:orange", linewidth=0.8, alpha=0.85, label=f"OPC UA {var}")
        ax.set_ylabel(f"{var} (%)")
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9,
                  edgecolor="gray", fancybox=True)

    axes[0].set_title(
        f"Ventanas — SCADA (negro) vs OPC UA (naranja)\nEnsayo: {label}",
        fontsize=10,
    )
    axes[-1].set_xlabel("Fecha (UTC)")
    axes[-1].xaxis.set_major_formatter(formatter)
    axes[-1].xaxis.set_major_locator(locator)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130, bbox_inches="tight")
    logger.info(f"  Guardada: {output}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def comparar_mes(mes: str, df_s: pd.DataFrame, df_o: pd.DataFrame, output: Path, show: bool = True):
    inicio = pd.Timestamp(mes + "-01")
    fin    = (inicio + pd.offsets.MonthEnd(0)).replace(hour=23, minute=59, second=59)

    s = df_s.loc[inicio:fin]
    o = df_o.loc[inicio:fin]

    if len(s) == 0 and len(o) == 0:
        logger.warning(f"Sin datos para {mes} en ninguna fuente — omitiendo.")
        return

    # Métricas
    idx_comun = s.index.intersection(o.index)
    print(f"\n--- {mes} ---")
    for var in VARIABLES_VENTANA:
        if var not in s.columns or var not in o.columns:
            continue
        ref  = s.loc[idx_comun, var]
        cand = o.loc[idx_comun, var]
        mask = ref.notna() & cand.notna()
        if mask.sum() < 100:
            print(f"  {var}: sin datos suficientes")
            continue
        mae  = (ref[mask] - cand[mask]).abs().mean()
        corr = ref[mask].corr(cand[mask])
        mean_opc = cand[mask].mean()
        print(f"  {var}: MAE={mae:.2f}  corr={corr:.4f}  media_OPC={mean_opc:.1f}")

    # Gráfica: una fila por variable
    n = len(VARIABLES_VENTANA)
    fig, axes = plt.subplots(n, 1, figsize=(16, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, var in zip(axes, VARIABLES_VENTANA):
        if var in s.columns:
            ax.plot(s.index.to_numpy(), s[var].to_numpy(),
                    color="black", linewidth=0.8, alpha=0.8, label=f"SCADA {var}")
        if var in o.columns:
            ax.plot(o.index.to_numpy(), o[var].to_numpy(),
                    color="tab:orange", linewidth=0.7, alpha=0.8, label=f"OPC UA {var}")
        ax.set_ylabel(f"{var} (%)")
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9,
                  edgecolor="gray", fancybox=True)

    axes[0].set_title(f"Ventanas — SCADA (negro) vs OPC UA (naranja) — {mes}", fontsize=10)
    axes[-1].set_xlabel("Fecha (UTC)")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=1))
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130, bbox_inches="tight")
    logger.info(f"  Guardada: {output}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compara UVENT SCADA vs OPC UA mes a mes")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mes",     help="Mes concreto en formato YYYY-MM")
    group.add_argument("--loop",    action="store_true", help="Todos los meses con OPC UA disponible")
    group.add_argument("--ensayos", action="store_true", help="Solo períodos de ensayo de ventilación (lee el Excel de ensayos)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "ventanas_meses",
    )
    args = parser.parse_args()

    if not CSV_SCADA.exists():
        logger.error(f"CSV SCADA no encontrado: {CSV_SCADA}")
        logger.error("Genera con: python src/prepare_dataset.py --source scada")
        sys.exit(1)
    if not CSV_OPCUA.exists():
        logger.error(f"CSV OPC UA no encontrado: {CSV_OPCUA}")
        logger.error("Genera con: python src/prepare_dataset.py --source opcua")
        sys.exit(1)

    logger.info("Cargando CSVs...")
    df_s = _cargar_csv(CSV_SCADA)
    df_o = _cargar_csv(CSV_OPCUA)
    logger.info(f"  SCADA:  {len(df_s):,} filas")
    logger.info(f"  OPC UA: {len(df_o):,} filas")

    if args.mes:
        output = PROJECT_ROOT / "data" / "comparacion_ventanas_opcua.png"
        comparar_mes(args.mes, df_s, df_o, output, show=True)
    elif args.ensayos:
        periodos = _cargar_ensayos_ventilacion()
        if not periodos:
            sys.exit(1)
        out_dir = args.output_dir.parent / "ensayos_ventanas"
        logger.info(f"Períodos de ensayo de ventilación encontrados: {len(periodos)}")
        for inicio, fin, tipo in periodos:
            label  = f"{inicio.strftime('%Y-%m-%d')}_{fin.strftime('%Y-%m-%d')}"
            output = out_dir / f"ensayo_{label}.png"
            comparar_rango(
                label=f"{tipo}  ({inicio.strftime('%d/%m/%Y')} → {fin.strftime('%d/%m/%Y')})",
                inicio=inicio, fin=fin,
                df_s=df_s, df_o=df_o,
                output=output, show=False,
            )
        logger.info(f"\nFin. Gráficas en {out_dir}/")
    else:
        meses = _meses_disponibles(df_o)
        logger.info(f"Meses con OPC UA: {meses}")
        for mes in meses:
            output = args.output_dir / f"ventanas_{mes}.png"
            comparar_mes(mes, df_s, df_o, output, show=False)
        logger.info(f"\nFin. Gráficas en {args.output_dir}/")


if __name__ == "__main__":
    main()
