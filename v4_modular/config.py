"""
config.py — Configuración central de v4_modular
================================================
Todas las constantes del proyecto están aquí: rutas, columnas, hiperparámetros.
Si necesitas cambiar algo, solo lo tocas en este fichero y afecta a todos los notebooks.
"""

import os

# ─────────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────────

# Directorio raíz del proyecto (donde está este config.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Datos de entrada (CSV original)
DATA_RAW_DIR   = os.path.join(BASE_DIR, '..', 'data')          # carpeta con los CSV originales
DATA_INTERIM   = os.path.join(BASE_DIR, 'data', 'interim')     # DataFrames intermedios (.parquet)
DATA_MODELS    = os.path.join(BASE_DIR, 'data', 'models')      # Modelos entrenados (.joblib)
DATA_PLOTS     = os.path.join(BASE_DIR, 'data', 'plots')       # Gráficas generadas (.png)

# Nombre del fichero combinado SCADA + OPC UA
DATASET_NAME   = 'combined_2024_03_06-2025_11_30_1min'
DATASET_PATH   = os.path.join(DATA_RAW_DIR, f'{DATASET_NAME}.csv')

# Período de análisis (1 año completo de datos combinados)
FECHA_INICIO   = '2024-03-06'
FECHA_FIN      = '2025-03-07'

# ─────────────────────────────────────────────
# COLUMNAS DEL DATASET
# ─────────────────────────────────────────────

# Sensores y actuadores usados en el modelo
# Nota: UVENT_cen y UVENT_lN excluidos (señal _POS/_POS_VALOR no refleja posición real,
#       MAE típico 40-85% vs encoder físico)
COLUMNAS_SENSORES = [
    'PCO2EXT', 'PHEXT', 'PRAD', 'PRGINT', 'PTEXT', 'PVV',
    'XCO2I', 'XHINV', 'XTINV', 'XTS'
]

# Columnas a excluir de las features de entrada a los modelos ML
COLUMNAS_EXCLUIR_FEATURES = [
    'Fecha', 'etiqueta_deteccion', 'etiqueta_tipo_anomalia',
    'UVENT_cen', 'UVENT_lN'   # excluidas por señal no fiable
]

# Etiquetas de anomalías
ETIQUETA_NORMAL    = 'normal'
ETIQUETA_ANOMALIA  = 'anomalia'

TIPOS_ANOMALIA = [
    'Datos Faltantes',
    'Sensor Atascado',
    'Ruido',
    'Valores Fuera de Rango',
    'Desviación de Correlación',
    'Contextual'
]

# ─────────────────────────────────────────────
# RANGOS FÍSICOS DE HARDWARE (v3 — M3)
# Fuente: AGROCONNECT_Variables_PLCs.xlsx
# ─────────────────────────────────────────────

RANGOS_FISICOS = {
    'PCO2EXT': {'min':   0.0, 'max': 2000.0},   # E+E EE820: 0-2000 ppm
    'PHEXT':   {'min':   0.0, 'max':  100.0},   # Campbell HC2S3: 0-100 %HR
    'PRAD':    {'min':   0.0, 'max': 1500.0},   # Kipp & Zonen CMP3: 0-1500 W/m²
    'PRGINT':  {'min':   0.0, 'max': 1200.0},   # Apogee SQ-500: 0-1200 W/m²
    'PTEXT':   {'min': -20.0, 'max':   60.0},   # PT100 clase A: -20..+60 °C
    'PVV':     {'min':   0.0, 'max':   50.0},   # Aanderaa 2740: 0-50 m/s
    'XCO2I':   {'min': 300.0, 'max': 3000.0},   # GMP343: 300-3000 ppm
    'XHINV':   {'min':   0.0, 'max':  100.0},   # Sensirion SHT31: 0-100 %HR
    'XTINV':   {'min':  -5.0, 'max':   60.0},   # Sensirion SHT31: -5..+60 °C
    'XTS':     {'min':  -5.0, 'max':   70.0},   # Pt100 suelo: -5..+70 °C
}

# ─────────────────────────────────────────────
# PARÁMETROS DE INYECCIÓN DE ANOMALÍAS
# ─────────────────────────────────────────────

INYECCION = {
    'datos_faltantes': {
        'porcentaje_filas': 0.02,        # 2% de filas con NaN
        'num_sensores_por_fila': 1,
    },
    'sensor_atascado': {
        'num_secuencias': 150,           # subido de 50 → 150 para mejorar recall (era 56.3%)
        'duracion_min': 5,               # mínimo 5 minutos
        'duracion_max': 20,              # máximo 20 minutos
    },
    'ruido': {
        'porcentaje_filas': 0.03,        # 3% de filas con ruido
        'factor_std_min': 3,
        'factor_std_max': 5,
        'num_sensores_por_fila': 1,
    },
    'fuera_rango': {
        'porcentaje_filas': 0.02,        # 2% de filas fuera de rango
        'num_sensores_por_fila': 1,
        'margen_error_factor': 0.1,
    },
    'desviacion_correlacion': {
        'porcentaje_filas': 0.015,       # 1.5% de filas
        'filtro_hora_inicio': 6,
        'filtro_hora_fin': 20,
    },
    'contextual_luz': {
        'porcentaje_filas': 0.01,        # 1% con luz nocturna anómala
        'hora_noche_inicio': 23,
        'hora_noche_fin': 4,
        'prad_umbral_cero': 1.0,
        'prgint_min': 20.0,
        'prgint_max': 50.0,
    },
    'contextual_co2': {
        'num_secuencias': 90,            # subido de 30 → 90 para equilibrar muestras
        'umbral_diff_co2': 30.0,         # v3 M9: condición relajada (era 50 ppm en v2)
        'duracion_min': 3,               # v3 M9: 3-8 min (era 5-15 min en v2)
        'duracion_max': 8,
    },
}

# ─────────────────────────────────────────────
# PARÁMETROS DE FEATURE ENGINEERING (M1 + M2)
# ─────────────────────────────────────────────

FE_SENSORES = ['PCO2EXT', 'PHEXT', 'PRAD', 'PRGINT', 'PTEXT', 'PVV',
               'XCO2I', 'XHINV', 'XTINV', 'XTS']

FE_VENTANAS = {
    'rolling_30m': 30,     # ventana 30 minutos
    'rolling_3h': 180,     # ventana 3 horas
}

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

# TimeSeriesSplit (v3 M6)
TSCV_N_SPLITS = 4

# ─────────────────────────────────────────────
# PARÁMETROS DE CORRECCIÓN
# ─────────────────────────────────────────────

FACTOR_UMBRAL_CORRECCION = 2.0   # desviaciones estándar para umbral de interpolación

IMPUTER_RF_PARAMS = {            # RandomForestRegressor dentro de IterativeImputer
    'n_estimators': 30,
    'random_state': 42,
    'max_depth':    10,
    'min_samples_leaf': 5,
    'n_jobs': -1,
}
IMPUTER_MAX_ITER = 10

# ─────────────────────────────────────────────
# MEJORAS PENDIENTES (v5)
# ─────────────────────────────────────────────

# [v5] Features estacionales — añadir si los resultados muestran falsos positivos
# en épocas concretas (ej. CO2 alto en invierno por ventanas cerradas, temp baja en enero).
# 'Mes' ya captura esto implícitamente, pero una codificación explícita puede ayudar.
#
# Opción A — one-hot (simple):
#   df['es_verano']   = df['Mes'].isin([6, 7, 8]).astype(int)
#   df['es_invierno'] = df['Mes'].isin([12, 1, 2]).astype(int)
#
# Opción B — cíclica (recomendada, no rompe la continuidad diciembre→enero):
#   df['mes_sin'] = np.sin(2 * np.pi * df['Mes'] / 12)
#   df['mes_cos'] = np.cos(2 * np.pi * df['Mes'] / 12)
#   df['hora_sin'] = np.sin(2 * np.pi * df['Hora'] / 24)
#   df['hora_cos'] = np.cos(2 * np.pi * df['Hora'] / 24)
#
# Añadir en 03_features.ipynb tras las features rolling (M1).

# [v5] Modelo único multiclase — eliminar cascada M1→M2, reducir errores propagados
# [v5] LightGBM como estimador del IterativeImputer — x5-10 más rápido que RF
# [v5] columna 'ensayo' como feature one-hot — días de experimento tienen patrones distintos

# ─────────────────────────────────────────────
# RUTAS DE FICHEROS DE SALIDA
# ─────────────────────────────────────────────

# DataFrames intermedios
PARQUET_01 = os.path.join(DATA_INTERIM, '01_datos_cargados.parquet')
PARQUET_02 = os.path.join(DATA_INTERIM, '02_datos_inyectados.parquet')
PARQUET_03 = os.path.join(DATA_INTERIM, '03_datos_features.parquet')
PARQUET_04 = os.path.join(DATA_INTERIM, '04_modelo1_predicciones.parquet')
PARQUET_06 = os.path.join(DATA_INTERIM, '06_datos_corregidos.parquet')

# Modelos entrenados
MODELO_1_PATH       = os.path.join(DATA_MODELS, 'modelo_1_detector.joblib')
IMPUTER_M1_PATH     = os.path.join(DATA_MODELS, 'imputer_modelo_1.joblib')
FEATURES_M1_PATH    = os.path.join(DATA_MODELS, 'features_modelo_1.joblib')
MODELO_2_PATH       = os.path.join(DATA_MODELS, 'modelo_2_clasificador.joblib')
LABEL_ENC_M2_PATH   = os.path.join(DATA_MODELS, 'label_encoder_modelo_2.joblib')
IMPUTER_FALT_PATH   = os.path.join(DATA_MODELS, 'imputer_datos_faltantes.joblib')
