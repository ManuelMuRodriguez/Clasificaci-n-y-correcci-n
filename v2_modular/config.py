"""
config.py — Configuración central de v2_modular
================================================
Todas las constantes del proyecto están aquí: rutas, columnas, hiperparámetros.
Si necesitas cambiar algo, solo lo tocas en este fichero y afecta a todos los notebooks.

Diferencias respecto a v4_modular (v3):
- UVENT_cen y UVENT_lN incluidos en el modelo
- Rangos de anomalías por percentiles estadísticos (P0.1/P99.9), no por hardware físico
- Split aleatorio 70/30 (sin TimeSeriesSplit)
- Sin features rolling ni cross-sensor (M1/M2)
- Sin umbrales adaptativos por CV (M4)
- Condición CO2 contextual más estricta (70%, 50 ppm, 5-15 min)
"""

import os

# ─────────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────────

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, '..', 'data')
DATA_INTERIM = os.path.join(BASE_DIR, 'data', 'interim')
DATA_MODELS  = os.path.join(BASE_DIR, 'data', 'models')
DATA_PLOTS   = os.path.join(BASE_DIR, 'data', 'plots')

DATASET_NAME = 'combined_2024_03_06-2025_11_30_1min'
DATASET_PATH = os.path.join(DATA_RAW_DIR, f'{DATASET_NAME}.csv')

FECHA_INICIO = '2024-03-06'
FECHA_FIN    = '2025-03-07'

# ─────────────────────────────────────────────
# COLUMNAS DEL DATASET
# ─────────────────────────────────────────────

# v2: UVENT_cen y UVENT_lN incluidos (a diferencia de v3 donde se excluyeron)
COLUMNAS_SENSORES = [
    'PCO2EXT', 'PHEXT', 'PRAD', 'PRGINT', 'PTEXT', 'PVV',
    'XCO2I', 'XHINV', 'XTINV', 'XTS',
    'UVENT_cen', 'UVENT_lN'
]

# Columnas a excluir de las features de entrada a los modelos ML
COLUMNAS_EXCLUIR_FEATURES = [
    'Fecha', 'etiqueta_deteccion', 'etiqueta_tipo_anomalia'
]

ETIQUETA_NORMAL   = 'normal'
ETIQUETA_ANOMALIA = 'anomalia'

TIPOS_ANOMALIA = [
    'Datos Faltantes',
    'Sensor Atascado',
    'Ruido',
    'Valores Fuera de Rango',
    'Desviación de Correlación',
    'Contextual'
]

# ─────────────────────────────────────────────
# RANGOS DE ANOMALÍAS — v2: PERCENTILES P0.1 / P99.9
# Se calculan automáticamente desde df_original en 02_inyeccion_anomalias.ipynb
# Este diccionario se rellena en ejecución, aquí solo se define la estructura
# ─────────────────────────────────────────────

# Percentiles usados para definir "fuera de rango" en v2
PERCENTIL_INFERIOR = 0.001   # P0.1
PERCENTIL_SUPERIOR = 0.999   # P99.9

# ─────────────────────────────────────────────
# PARÁMETROS DE INYECCIÓN DE ANOMALÍAS
# ─────────────────────────────────────────────

INYECCION = {
    'datos_faltantes': {
        'porcentaje_filas': 0.02,
        'num_sensores_por_fila': 1,
    },
    'sensor_atascado': {
        'num_secuencias': 150,           # subido de 50 → 150 para mejorar recall (era 56.3%)
        'duracion_min': 5,
        'duracion_max': 20,
    },
    'ruido': {
        'porcentaje_filas': 0.03,
        'factor_std_min': 3,
        'factor_std_max': 5,
        'num_sensores_por_fila': 1,
    },
    'fuera_rango': {
        'porcentaje_filas': 0.02,
        'num_sensores_por_fila': 1,
        'margen_error_factor': 0.1,
    },
    'desviacion_correlacion': {
        'porcentaje_filas': 0.015,
        'filtro_hora_inicio': 6,
        'filtro_hora_fin': 20,
    },
    'contextual_luz': {
        'porcentaje_filas': 0.01,
        'hora_noche_inicio': 23,
        'hora_noche_fin': 4,
        'prad_umbral_cero': 1.0,
        'prgint_min': 20.0,
        'prgint_max': 50.0,
    },
    'contextual_co2': {
        'num_secuencias': 30,
        # v2: condición más estricta que v3 (v3 usa 30 ppm y 3-8 min)
        'umbral_vent_abierta': 70.0,   # ventanas deben estar >70% abiertas
        'umbral_diff_co2': 50.0,       # gradiente mínimo CO2 interior-exterior
        'duracion_min': 5,
        'duracion_max': 15,
    },
}

# ─────────────────────────────────────────────
# PARÁMETROS DE LOS PARES DE CORRELACIÓN
# v2: incluye UVENT_cen (excluido en v3)
# ─────────────────────────────────────────────

PARES_CORRELACION = [
    ('PRAD', 'PRGINT'),
    ('XTINV', 'XHINV'),
    ('PRAD', 'UVENT_cen'),   # v2: disponible; v3: sustituido por (PTEXT, XTINV)
]

COLUMNAS_PARA_PERCENTILES = ['PRAD', 'PRGINT', 'XTINV', 'XHINV', 'UVENT_cen']

# ─────────────────────────────────────────────
# HIPERPARÁMETROS DE MODELOS
# ─────────────────────────────────────────────

MODELO_1_PARAMS = {
    'n_estimators': 100,
    'random_state': 42,
    'class_weight': 'balanced',
    'n_jobs': -1,
}

MODELO_2_PARAMS = {
    'n_estimators': 100,
    'random_state': 42,
    'class_weight': 'balanced',
    'n_jobs': -1,
}

# v2: split aleatorio 70/30 (sin TimeSeriesSplit — diferencia principal con v3)
TRAIN_TEST_SPLIT = {
    'test_size': 0.3,
    'random_state': 42,
}

# ─────────────────────────────────────────────
# RUTAS DE FICHEROS DE SALIDA
# ─────────────────────────────────────────────

PARQUET_01 = os.path.join(DATA_INTERIM, '01_datos_cargados.parquet')
PARQUET_02 = os.path.join(DATA_INTERIM, '02_datos_inyectados.parquet')
PARQUET_03 = os.path.join(DATA_INTERIM, '03_datos_features.parquet')
PARQUET_04 = os.path.join(DATA_INTERIM, '04_modelo1_predicciones.parquet')
PARQUET_06 = os.path.join(DATA_INTERIM, '06_datos_corregidos.parquet')

MODELO_1_PATH     = os.path.join(DATA_MODELS, 'modelo_1_detector.joblib')
IMPUTER_M1_PATH   = os.path.join(DATA_MODELS, 'imputer_modelo_1.joblib')
FEATURES_M1_PATH  = os.path.join(DATA_MODELS, 'features_modelo_1.joblib')
MODELO_2_PATH     = os.path.join(DATA_MODELS, 'modelo_2_clasificador.joblib')
LABEL_ENC_M2_PATH = os.path.join(DATA_MODELS, 'label_encoder_modelo_2.joblib')
IMPUTER_FALT_PATH = os.path.join(DATA_MODELS, 'imputer_datos_faltantes.joblib')
