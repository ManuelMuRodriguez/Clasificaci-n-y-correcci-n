"""
prepare_dataset.py
==================
Genera el CSV de entrada para el sistema de detección y corrección de anomalías.

Pipeline:
  1. Carga todos los XLSX de la carpeta Dataset usando data_loader
  2. Filtra el rango útil (desde 2023-12-13, primer día con sensores exteriores)
  3. Selecciona las 12 variables del paper y las renombra
  4. Agrega las ventanas de ventilación (media de posiciones reales)
  5. Resamplea de 30 s a 5 min (media)
  6. Exporta a CSV con el formato esperado por el notebook

Mapeo de columnas
-----------------
  PCO2EXT  ← CO2_EXTERIOR_10M
  PHEXT    ← HR_EXTERIOR_10M
  PRAD     ← RADGLOBAL_EXTERIOR_10M
  PRGINT   ← INVER_RADGLOBAL_INTERIOR_S1
  PTEXT    ← TEMP_EXTERIOR_10M
  PVV      ← VELVIENTO_EXTERIOR_10M
  XCO2I    ← INVER_CO2_INTERIOR_S1
  XHINV    ← INVER_HR_INTERIOR_S1
  XTINV    ← INVER_TEMP_INTERIOR_S1
  XTS      ← INVER_TEMP_SUELO5_S1
  UVENT_cen ← media de UVCEN1_1_POS, UVCEN1_2_POS, UVCEN1_3_POS,
                         UVCEN2_1_POS, UVCEN2_2_POS, UVCEN2_3_POS
  UVENT_lN  ← media de UVLAT1N_POS, UVLAT1ON_POS, UVLAT1OS_POS, UVLAT1S_POS,
                         UVLAT2E_POS, UVLAT2N_POS, UVLAT2S_POS

Uso
---
  python src/prepare_dataset.py
  python src/prepare_dataset.py --dataset-dir Dataset --output data.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Añadir raíz del proyecto al path para importar src.data_loader
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_all_files

# ─────────────────────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Fecha de inicio: primer día con todos los sensores exteriores activos
FECHA_INICIO = "2023-12-13"

# Columnas de ventanas centrales (posición real, 0-100 %)
COLS_UVENT_CEN = [
    "UVCEN1_1_POS", "UVCEN1_2_POS", "UVCEN1_3_POS",
    "UVCEN2_1_POS", "UVCEN2_2_POS", "UVCEN2_3_POS",
]

# Columnas de ventanas laterales (posición real, 0-100 %)
COLS_UVENT_LN = [
    "UVLAT1N_POS", "UVLAT1ON_POS", "UVLAT1OS_POS", "UVLAT1S_POS",
    "UVLAT2E_POS", "UVLAT2N_POS", "UVLAT2S_POS",
]

# Mapeo: columna original → nombre del paper
COLUMN_MAP = {
    "CO2_EXTERIOR_10M":           "PCO2EXT",
    "HR_EXTERIOR_10M":            "PHEXT",
    "RADGLOBAL_EXTERIOR_10M":     "PRAD",
    "INVER_RADGLOBAL_INTERIOR_S1":"PRGINT",
    "TEMP_EXTERIOR_10M":          "PTEXT",
    "VELVIENTO_EXTERIOR_10M":     "PVV",
    "INVER_CO2_INTERIOR_S1":      "XCO2I",
    "INVER_HR_INTERIOR_S1":       "XHINV",
    "INVER_TEMP_INTERIOR_S1":     "XTINV",
    "INVER_TEMP_SUELO5_S1":       "XTS",
}

# Orden final de columnas en el CSV
COLS_FINALES = [
    "Fecha",
    "PCO2EXT", "PHEXT", "PRAD", "PRGINT", "PTEXT", "PVV",
    "UVENT_cen", "UVENT_lN",
    "XCO2I", "XHINV", "XTINV", "XTS",
]


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def preparar_dataset(dataset_dir: Path, output_path: Path) -> pd.DataFrame:

    # 1. Cargar todos los xlsx
    logger.info("Paso 1/5 — Cargando ficheros xlsx...")
    df = load_all_files(dataset_dir=dataset_dir, show_progress=True)
    logger.info(f"  Raw: {len(df):,} filas × {len(df.columns)} columnas")

    # 2. Filtrar rango útil
    logger.info(f"Paso 2/5 — Filtrando desde {FECHA_INICIO}...")
    df["FECHA"] = pd.to_datetime(df["FECHA"])
    df = df[df["FECHA"] >= FECHA_INICIO].copy()
    logger.info(f"  Tras filtro: {len(df):,} filas")

    # 3. Calcular ventilación agregada (media de posiciones reales)
    logger.info("Paso 3/5 — Agregando ventanas de ventilación...")
    cols_cen_presentes = [c for c in COLS_UVENT_CEN if c in df.columns]
    cols_ln_presentes  = [c for c in COLS_UVENT_LN  if c in df.columns]

    if not cols_cen_presentes:
        logger.warning("  No se encontraron columnas de ventanas centrales (_POS). UVENT_cen = NaN")
        df["UVENT_cen"] = float("nan")
    else:
        df["UVENT_cen"] = df[cols_cen_presentes].mean(axis=1)
        logger.info(f"  UVENT_cen: media de {len(cols_cen_presentes)} columnas → {cols_cen_presentes}")

    if not cols_ln_presentes:
        logger.warning("  No se encontraron columnas de ventanas laterales (_POS). UVENT_lN = NaN")
        df["UVENT_lN"] = float("nan")
    else:
        df["UVENT_lN"] = df[cols_ln_presentes].mean(axis=1)
        logger.info(f"  UVENT_lN:  media de {len(cols_ln_presentes)} columnas → {cols_ln_presentes}")

    # 4. Seleccionar y renombrar columnas
    logger.info("Paso 4/5 — Seleccionando y renombrando columnas...")
    cols_seleccionar = list(COLUMN_MAP.keys()) + ["UVENT_cen", "UVENT_lN", "FECHA"]
    cols_disponibles = [c for c in cols_seleccionar if c in df.columns]
    cols_faltantes   = [c for c in cols_seleccionar if c not in df.columns]

    if cols_faltantes:
        logger.warning(f"  Columnas no encontradas en el dataset: {cols_faltantes}")

    df = df[cols_disponibles].copy()
    df = df.rename(columns=COLUMN_MAP)
    df = df.rename(columns={"FECHA": "Fecha"})

    # 5. Resamplear de 30 s a 5 min
    logger.info("Paso 5/5 — Resampleando de 30 s a 5 min (media)...")
    df = df.set_index("Fecha")
    df = df.resample("5min").mean()
    df = df.reset_index()
    logger.info(f"  Tras resample: {len(df):,} filas")

    # Ordenar columnas según el orden final esperado
    cols_presentes_final = [c for c in COLS_FINALES if c in df.columns]
    df = df[cols_presentes_final]

    # Formatear fecha como string (formato compatible con el notebook)
    df["Fecha"] = df["Fecha"].dt.strftime("%d/%m/%Y %H:%M:%S")

    # 6. Exportar CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, decimal=".")
    logger.info(f"\nCSV guardado en: {output_path}")
    logger.info(f"  Filas:    {len(df):,}")
    logger.info(f"  Columnas: {list(df.columns)}")
    logger.info(f"  Período:  {df['Fecha'].iloc[0]}  →  {df['Fecha'].iloc[-1]}")

    # Resumen de nulos
    nulos = df.isnull().sum()
    nulos_pct = (nulos / len(df) * 100).round(2)
    resumen = pd.DataFrame({"nulos": nulos, "pct": nulos_pct})
    resumen = resumen[resumen["nulos"] > 0]
    if not resumen.empty:
        logger.info(f"\nColumnas con valores nulos:\n{resumen.to_string()}")
    else:
        logger.info("\nSin valores nulos en el CSV final.")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Genera el CSV de entrada para el sistema de anomalías.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "Dataset",
        help="Carpeta con los ficheros AGROCONNECT_*.xlsx (default: Dataset/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "2023_12_13-2024_12_31.csv",
        help="Ruta del CSV de salida (default: data/2023_12_13-2024_12_31.csv)",
    )
    args = parser.parse_args()

    preparar_dataset(dataset_dir=args.dataset_dir, output_path=args.output)


if __name__ == "__main__":
    main()
