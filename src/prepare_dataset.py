"""
prepare_dataset.py
==================
Genera el CSV de entrada para el sistema de detección y corrección de anomalías.
Soporta dos fuentes de datos: SCADA (AGROCONNECT XLSX) y OPC UA (TXT).

Pipeline común
--------------
  1. Carga todos los ficheros de la fuente seleccionada
  2. Filtra el rango útil
  3. Calcula UVENT_cen y UVENT_lN como media de posiciones reales
  4. Selecciona las 12 variables del paper y las renombra
  5. Resamplea a 1 min (media)
  6. Exporta CSV

Mapeo de columnas — SCADA
--------------------------
  PCO2EXT   ← CO2_EXTERIOR_10M
  PHEXT     ← HR_EXTERIOR_10M
  PRAD      ← RADGLOBAL_EXTERIOR_10M
  PRGINT    ← INVER_RADGLOBAL_INTERIOR_S1
  PTEXT     ← TEMP_EXTERIOR_10M
  PVV       ← VELVIENTO_EXTERIOR_10M
  XCO2I     ← INVER_CO2_INTERIOR_S1
  XHINV     ← INVER_HR_INTERIOR_S1
  XTINV     ← INVER_TEMP_INTERIOR_S1
  XTS       ← INVER_TEMP_SUELO5_S1
  UVENT_cen ← media de UVCEN1_1_POS … UVCEN2_3_POS
  UVENT_lN  ← media de UVLAT1N_POS … UVLAT2S_POS

Mapeo de columnas — OPC UA (mismas variables, prefijo OPC_)
------------------------------------------------------------
  PCO2EXT   ← OPC_CO2_EXTERIOR_10M
  PHEXT     ← OPC_HR_EXTERIOR_10M
  PRAD      ← OPC_RADGLOBAL_EXTERIOR_10M
  PRGINT    ← OPC_INVER_RADGLOBAL_INTERIOR_S1
  PTEXT     ← OPC_TEMP_EXTERIOR_10M
  PVV       ← OPC_VELVIENTO_EXTERIOR_10M
  XCO2I     ← OPC_INVER_CO2_INTERIOR_S1
  XHINV     ← OPC_INVER_HR_INTERIOR_S1
  XTINV     ← OPC_INVER_TEMP_INTERIOR_S1
  XTS       ← OPC_INVER_TEMP_SUELO5_S1  (verificado: S1 OPC UA = S1 SCADA, MAE=0.017°C)
  UVENT_cen ← media de OPC_UVCEN1_1_POS_VALOR … OPC_UVCEN2_3_POS_VALOR  (no _POS, que es binario 0/100)
  UVENT_lN  ← media de OPC_UVLAT1N_POS_VALOR … OPC_UVLAT2S_POS_VALOR

Uso
---
  # SCADA (por defecto)
  python src/prepare_dataset.py
  python src/prepare_dataset.py --source scada

  # OPC UA
  python src/prepare_dataset.py --source opcua
  python src/prepare_dataset.py --source opcua --output data/opcua_2024_07-2025_03_1min.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_all_files, load_all_opcua_files

# ─────────────────────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

FECHA_INICIO = "2024-03-06"  # Primer día completo con datos de ventanas (_POS)
FECHA_FIN    = "2025-11-30"  # Último día con datos SCADA disponibles

# ── SCADA ────────────────────────────────────────────────────────────────────

COLS_UVENT_CEN_SCADA = [
    "UVCEN1_1_POS", "UVCEN1_2_POS", "UVCEN1_3_POS",
    "UVCEN2_1_POS", "UVCEN2_2_POS", "UVCEN2_3_POS",
]
COLS_UVENT_LN_SCADA = [
    "UVLAT1N_POS", "UVLAT1ON_POS", "UVLAT1OS_POS", "UVLAT1S_POS",
    "UVLAT2E_POS", "UVLAT2N_POS", "UVLAT2S_POS",
]
COLUMN_MAP_SCADA = {
    "CO2_EXTERIOR_10M":            "PCO2EXT",
    "HR_EXTERIOR_10M":             "PHEXT",
    "RADGLOBAL_EXTERIOR_10M":      "PRAD",
    "INVER_RADGLOBAL_INTERIOR_S1": "PRGINT",
    "TEMP_EXTERIOR_10M":           "PTEXT",
    "VELVIENTO_EXTERIOR_10M":      "PVV",
    "INVER_CO2_INTERIOR_S1":       "XCO2I",
    "INVER_HR_INTERIOR_S1":        "XHINV",
    "INVER_TEMP_INTERIOR_S1":      "XTINV",
    "INVER_TEMP_SUELO5_S1":        "XTS",
}

# ── OPC UA ───────────────────────────────────────────────────────────────────

COLS_UVENT_CEN_OPCUA = [
    "OPC_UVCEN1_1_POS_VALOR", "OPC_UVCEN1_2_POS_VALOR", "OPC_UVCEN1_3_POS_VALOR",
    "OPC_UVCEN2_1_POS_VALOR", "OPC_UVCEN2_2_POS_VALOR", "OPC_UVCEN2_3_POS_VALOR",
]
COLS_UVENT_LN_OPCUA = [
    "OPC_UVLAT1N_POS_VALOR", "OPC_UVLAT1ON_POS_VALOR", "OPC_UVLAT1OS_POS_VALOR", "OPC_UVLAT1S_POS_VALOR",
    "OPC_UVLAT2E_POS_VALOR", "OPC_UVLAT2N_POS_VALOR",  "OPC_UVLAT2S_POS_VALOR",
]
COLUMN_MAP_OPCUA = {
    "OPC_CO2_EXTERIOR_10M":            "PCO2EXT",
    "OPC_HR_EXTERIOR_10M":             "PHEXT",
    "OPC_RADGLOBAL_EXTERIOR_10M":      "PRAD",
    "OPC_INVER_RADGLOBAL_INTERIOR_S1": "PRGINT",
    "OPC_TEMP_EXTERIOR_10M":           "PTEXT",
    "OPC_VELVIENTO_EXTERIOR_10M":      "PVV",
    "OPC_INVER_CO2_INTERIOR_S1":       "XCO2I",
    "OPC_INVER_HR_INTERIOR_S1":        "XHINV",
    "OPC_INVER_TEMP_INTERIOR_S1":      "XTINV",
    "OPC_INVER_TEMP_SUELO5_S1":        "XTS",
}

# ── Común ────────────────────────────────────────────────────────────────────

VARIABLES_PAPER = [
    "PCO2EXT", "PHEXT", "PRAD", "PRGINT", "PTEXT", "PVV",
    "UVENT_cen", "UVENT_lN", "XCO2I", "XHINV", "XTINV", "XTS",
]

# Períodos donde OPC UA tiene datos incorrectos para una variable concreta.
# En estos rangos NO se usa OPC UA para rellenar NaN de SCADA.
# Formato: {variable: [(inicio, fin), ...]}   — ambos extremos inclusive, UTC.
OPCUA_PERIODOS_EXCLUIDOS: dict[str, list[tuple[str, str]]] = {
    # XTS (INVER_TEMP_SUELO5_S1): el sensor OPC UA estaba mal conectado en julio
    # y los primeros días de agosto de 2024. Verificado visualmente con comparar_xts_opcua.py.
    "XTS": [("2024-07-01", "2024-08-04 23:59:59")],
}

COLS_FINALES = [
    "Fecha",
    "PCO2EXT", "PHEXT", "PRAD", "PRGINT", "PTEXT", "PVV",
    "UVENT_cen", "UVENT_lN",
    "XCO2I", "XHINV", "XTINV", "XTS",
    "ensayo",
]

# Ensayos registrados — fuente: Dataset/Ensayos/Lista de ensayos Invernadero AgroConnect.xlsx
# Formato: (fecha_inicio, fecha_fin, tipo)  — ambos extremos inclusive (día completo UTC)
# Tipos: "ventilacion", "calefaccion", "deshumidificacion", "co2"
ENSAYOS: list[tuple[str, str, str]] = [
    # ── 2024 ──────────────────────────────────────────────────────────────────
    ("2024-03-18", "2024-03-25", "ventilacion"),
    ("2024-04-01", "2024-04-05", "ventilacion"),
    ("2024-04-23", "2024-04-24", "ventilacion"),
    ("2024-11-18", "2024-11-24", "ventilacion"),
    # ── 2025 ──────────────────────────────────────────────────────────────────
    ("2025-01-13", "2025-02-26", "calefaccion"),
    ("2025-03-13", "2025-03-27", "deshumidificacion"),
    ("2025-04-04", "2025-04-04", "calefaccion"),
    ("2025-04-19", "2025-04-19", "ventilacion"),
    ("2025-06-12", "2025-06-24", "ventilacion"),
    ("2025-07-22", "2025-07-22", "co2"),
    ("2025-09-17", "2025-09-17", "ventilacion"),
    ("2025-09-24", "2025-09-24", "ventilacion"),
    ("2025-09-30", "2025-09-30", "ventilacion"),
    ("2025-11-19", "2025-11-20", "ventilacion"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _agregar_columna_ensayo(df: pd.DataFrame) -> pd.DataFrame:
    """Añade columna 'ensayo' con el tipo de ensayo activo en cada timestamp (o vacío)."""
    df["ensayo"] = ""
    fecha = pd.to_datetime(df["Fecha"], dayfirst=True, errors="coerce")
    for inicio, fin, tipo in ENSAYOS:
        t_ini = pd.Timestamp(inicio)
        t_fin = pd.Timestamp(fin + " 23:59:59")
        mask = (fecha >= t_ini) & (fecha <= t_fin)
        df.loc[mask, "ensayo"] = tipo
    n = (df["ensayo"] != "").sum()
    logger.info(f"  Columna 'ensayo': {n:,} registros marcados con tipo de ensayo")
    return df


def _agregar_ventilacion(df, cols_cen, cols_ln):
    cols_cen_ok = [c for c in cols_cen if c in df.columns]
    cols_ln_ok  = [c for c in cols_ln  if c in df.columns]

    if not cols_cen_ok:
        logger.warning("  No se encontraron columnas de ventanas centrales. UVENT_cen = NaN")
        df["UVENT_cen"] = float("nan")
    else:
        df["UVENT_cen"] = df[cols_cen_ok].mean(axis=1)
        logger.info(f"  UVENT_cen: media de {len(cols_cen_ok)} columnas")

    if not cols_ln_ok:
        logger.warning("  No se encontraron columnas de ventanas laterales. UVENT_lN = NaN")
        df["UVENT_lN"] = float("nan")
    else:
        df["UVENT_lN"] = df[cols_ln_ok].mean(axis=1)
        logger.info(f"  UVENT_lN:  media de {len(cols_ln_ok)} columnas")

    return df


def preparar_dataset_scada(dataset_dir: Path, output_path: Path) -> pd.DataFrame:
    logger.info("=== Fuente: SCADA (AGROCONNECT XLSX) ===")

    logger.info("Paso 1/5 — Cargando ficheros xlsx...")
    df = load_all_files(dataset_dir=dataset_dir / "SCADA", show_progress=True)
    logger.info(f"  Raw: {len(df):,} filas × {len(df.columns)} columnas")

    logger.info(f"Paso 2/5 — Filtrando {FECHA_INICIO} → {FECHA_FIN}...")
    df["FECHA"] = pd.to_datetime(df["FECHA"])
    df = df[(df["FECHA"] >= FECHA_INICIO) & (df["FECHA"] <= FECHA_FIN)].copy()
    logger.info(f"  Tras filtro: {len(df):,} filas")

    logger.info("Paso 3/5 — Agregando ventanas de ventilación...")
    df = _agregar_ventilacion(df, COLS_UVENT_CEN_SCADA, COLS_UVENT_LN_SCADA)

    logger.info("Paso 4/5 — Seleccionando y renombrando columnas...")
    cols_sel = list(COLUMN_MAP_SCADA.keys()) + ["UVENT_cen", "UVENT_lN", "FECHA"]
    cols_ok  = [c for c in cols_sel if c in df.columns]
    cols_ko  = [c for c in cols_sel if c not in df.columns]
    if cols_ko:
        logger.warning(f"  Columnas no encontradas: {cols_ko}")
    df = df[cols_ok].copy()
    df = df.rename(columns=COLUMN_MAP_SCADA)
    df = df.rename(columns={"FECHA": "Fecha"})

    return _resamplear_y_exportar(df, output_path)


def preparar_dataset_opcua(opcua_dir: Path, output_path: Path) -> pd.DataFrame:
    logger.info("=== Fuente: OPC UA (TXT) ===")

    logger.info("Paso 1/5 — Cargando ficheros OPC UA...")
    df = load_all_opcua_files(opcua_dir=opcua_dir / "OPCUA", show_progress=True)
    logger.info(f"  Raw: {len(df):,} filas × {len(df.columns)} columnas")

    logger.info(f"Paso 2/5 — Filtrando {FECHA_INICIO} → {FECHA_FIN}...")
    df = df[(df["Fecha"] >= FECHA_INICIO) & (df["Fecha"] <= FECHA_FIN)].copy()
    logger.info(f"  Tras filtro: {len(df):,} filas")

    logger.info("Paso 3/5 — Agregando ventanas de ventilación...")
    df = _agregar_ventilacion(df, COLS_UVENT_CEN_OPCUA, COLS_UVENT_LN_OPCUA)

    logger.info("Paso 4/5 — Seleccionando y renombrando columnas...")
    cols_sel = list(COLUMN_MAP_OPCUA.keys()) + ["UVENT_cen", "UVENT_lN", "Fecha"]
    cols_ok  = [c for c in cols_sel if c in df.columns]
    cols_ko  = [c for c in cols_sel if c not in df.columns]
    if cols_ko:
        logger.warning(f"  Columnas no encontradas: {cols_ko}")
    df = df[cols_ok].copy()
    df = df.rename(columns=COLUMN_MAP_OPCUA)

    return _resamplear_y_exportar(df, output_path)


def preparar_dataset_combined(dataset_dir: Path, output_path: Path) -> pd.DataFrame:
    """
    Fusiona SCADA y OPC UA usando SCADA como base.
    Los NaN de SCADA se rellenan con el valor de OPC UA en el mismo minuto.
    Ambas fuentes se resamplean a 1 min antes de fusionar.
    """
    logger.info("=== Fuente: COMBINED (SCADA + OPC UA) ===")

    # Generar ambos datasets en memoria (sin exportar)
    logger.info("Cargando SCADA...")
    df_s = _preparar_sin_exportar_scada(dataset_dir)

    logger.info("Cargando OPC UA...")
    df_o = _preparar_sin_exportar_opcua(dataset_dir)

    # Fusionar: SCADA como base, rellenar NaN con OPC UA
    logger.info("Fusionando fuentes...")
    df_s = df_s.set_index("Fecha")
    df_o = df_o.set_index("Fecha")

    # Alinear al índice de SCADA y rellenar NaN
    df_o_aligned = df_o.reindex(df_s.index)
    df_combined = df_s.copy()
    for col in VARIABLES_PAPER:
        if col not in df_combined.columns or col not in df_o_aligned.columns:
            continue

        mask = df_combined[col].isna()

        # Excluir períodos donde OPC UA tiene datos incorrectos para esta variable
        if col in OPCUA_PERIODOS_EXCLUIDOS:
            for inicio_exc, fin_exc in OPCUA_PERIODOS_EXCLUIDOS[col]:
                periodo_malo = (df_combined.index >= inicio_exc) & (df_combined.index <= fin_exc)
                mask = mask & ~periodo_malo
                n_excluidos = (df_combined[col].isna() & periodo_malo).sum()
                if n_excluidos > 0:
                    logger.info(f"  {col}: excluidos {n_excluidos:,} minutos de OPC UA ({inicio_exc} → {fin_exc}, sensor mal conectado)")

        df_combined.loc[mask, col] = df_o_aligned.loc[mask, col]
        n_filled = mask.sum()
        if n_filled > 0:
            logger.info(f"  {col}: rellenados {n_filled:,} NaN con OPC UA")

    df_combined = df_combined.reset_index()

    # Resumen NaN tras fusión
    nulos = df_combined[VARIABLES_PAPER].isnull().sum()
    resumen = pd.DataFrame({"nulos": nulos, "pct": (nulos / len(df_combined) * 100).round(2)})
    resumen = resumen[resumen["nulos"] > 0]
    if not resumen.empty:
        logger.info(f"\nNaN restantes tras fusión:\n{resumen.to_string()}")
    else:
        logger.info("\nSin NaN en el dataset combinado.")

    return _exportar(df_combined, output_path)


def _preparar_sin_exportar_scada(dataset_dir: Path) -> pd.DataFrame:
    df = load_all_files(dataset_dir=dataset_dir / "SCADA", show_progress=True)
    df["FECHA"] = pd.to_datetime(df["FECHA"])
    df = df[(df["FECHA"] >= FECHA_INICIO) & (df["FECHA"] <= FECHA_FIN)].copy()
    df = _agregar_ventilacion(df, COLS_UVENT_CEN_SCADA, COLS_UVENT_LN_SCADA)
    cols_sel = list(COLUMN_MAP_SCADA.keys()) + ["UVENT_cen", "UVENT_lN", "FECHA"]
    df = df[[c for c in cols_sel if c in df.columns]].copy()
    df = df.rename(columns=COLUMN_MAP_SCADA)
    df = df.rename(columns={"FECHA": "Fecha"})
    df = df.set_index("Fecha")
    df = df.resample("1min").mean()
    return df.reset_index()


def _preparar_sin_exportar_opcua(dataset_dir: Path) -> pd.DataFrame:
    df = load_all_opcua_files(opcua_dir=dataset_dir / "OPCUA", show_progress=True)
    df = df[(df["Fecha"] >= FECHA_INICIO) & (df["Fecha"] <= FECHA_FIN)].copy()
    df = _agregar_ventilacion(df, COLS_UVENT_CEN_OPCUA, COLS_UVENT_LN_OPCUA)
    cols_sel = list(COLUMN_MAP_OPCUA.keys()) + ["UVENT_cen", "UVENT_lN", "Fecha"]
    df = df[[c for c in cols_sel if c in df.columns]].copy()
    df = df.rename(columns=COLUMN_MAP_OPCUA)
    df = df.set_index("Fecha")
    df = df.resample("1min").mean()
    return df.reset_index()


def _resamplear_y_exportar(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    logger.info("Paso 5/5 — Resampleando a 1 min (media)...")
    df = df.set_index("Fecha")
    df = df.resample("1min").mean()
    df = df.reset_index()
    logger.info(f"  Tras resample: {len(df):,} filas")
    return _exportar(df, output_path)


def _exportar(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    df = _agregar_columna_ensayo(df)
    cols_presentes = [c for c in COLS_FINALES if c in df.columns]
    df = df[cols_presentes]
    df["Fecha"] = df["Fecha"].dt.strftime("%d/%m/%Y %H:%M:%S")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, decimal=".")
    logger.info(f"\nCSV guardado en: {output_path}")
    logger.info(f"  Filas:    {len(df):,}")
    logger.info(f"  Columnas: {list(df.columns)}")
    logger.info(f"  Período:  {df['Fecha'].iloc[0]}  →  {df['Fecha'].iloc[-1]}")

    nulos = df.isnull().sum()
    resumen = pd.DataFrame({"nulos": nulos, "pct": (nulos / len(df) * 100).round(2)})
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
        "--source",
        choices=["scada", "opcua", "combined"],
        default="scada",
        help="Fuente de datos: 'scada', 'opcua' o 'combined' (SCADA + OPC UA fusionados). Default: scada",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "Dataset",
        help="Carpeta raíz del dataset (contiene SCADA/ y OPCUA/). Default: Dataset/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Ruta del CSV de salida. Si no se especifica, se genera automáticamente.",
    )
    args = parser.parse_args()

    if args.output is None:
        nombres = {"scada": "scada", "opcua": "opcua", "combined": "combined"}
        nombre = f"{nombres[args.source]}_2024_03_06-2025_11_30_1min.csv"
        args.output = PROJECT_ROOT / "data" / nombre

    if args.source == "scada":
        preparar_dataset_scada(dataset_dir=args.dataset_dir, output_path=args.output)
    elif args.source == "opcua":
        preparar_dataset_opcua(opcua_dir=args.dataset_dir, output_path=args.output)
    else:
        preparar_dataset_combined(dataset_dir=args.dataset_dir, output_path=args.output)


if __name__ == "__main__":
    main()
