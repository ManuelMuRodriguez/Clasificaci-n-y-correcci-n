# v2_modular — Pipeline de Detección y Corrección de Anomalías

Versión modular del pipeline v2. Mismo resultado que el notebook monolítico `Clasificación_y_Corrección_v2.ipynb`, reorganizado en notebooks independientes para facilitar la comprensión y depuración paso a paso.

---

## Estructura del proyecto

```
v2_modular/
├── config.py                      ← Configuración central (rutas, columnas, parámetros)
├── README.md                      ← Este fichero
│
├── 01_carga_datos.ipynb           ← Cargar CSV, filtrar fechas, añadir Hora/DiaSemana/Mes
├── 02_inyeccion_anomalias.ipynb   ← Inyectar 6 tipos de anomalías sintéticas
├── 03_features.ipynb              ← Selección de features (sin rolling ni cross-sensor)
├── 04_modelo_deteccion.ipynb      ← Modelo 1: detección binaria (normal/anomalía)
├── 05_modelo_clasificacion.ipynb  ← Modelo 2: clasificación de tipo de anomalía
├── 06_correccion.ipynb            ← Corrección por tipo (6 estrategias)
├── 07_evaluacion.ipynb            ← Evaluación con ground truth (datos sintéticos)
├── 08_inferencia_real.ipynb       ← Inferencia sobre datos reales del invernadero
│
└── data/
    ├── interim/                   ← DataFrames entre notebooks (.parquet)
    ├── models/                    ← Modelos entrenados (.joblib)
    └── plots/                     ← Gráficas generadas
```

---

## Cómo ejecutar

Ejecuta los notebooks en orden. Cada uno carga el parquet del anterior.

```
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08
```

---

## Cómo funciona la inyección de anomalías

El proceso de inyección es la base del entrenamiento supervisado. Sin él no habría ground truth y los modelos no podrían aprender. Funciona así:

1. **Se buscan filas** del dataset que cumplan ciertas condiciones (que sean 'normal', que tengan valor válido, que se den en un contexto específico, etc.)
2. **Se reemplaza el valor** de esa fila en la columna del sensor por un valor anómalo construido artificialmente
3. **Se marca la etiqueta** de esa fila como `'anomalia'` y se registra el tipo concreto

Cada tipo de anomalía tiene su propia estrategia de construcción del valor anómalo:

| Tipo | Qué se hace al valor |
|------|---------------------|
| **Datos Faltantes** | Se reemplaza por `NaN`. El más simple. |
| **Sensor Atascado** | Se coge el valor del punto de inicio y se **repite** durante 5-20 filas seguidas. No cambia el valor, lo congela en el tiempo. |
| **Ruido / Picos** | Se suma o resta entre 3 y 5 desviaciones estándar al valor original. Sigue siendo un número válido pero exagerado. |
| **Fuera de Rango** | Se reemplaza por un valor que esté por debajo del mínimo o por encima del máximo permitido para ese sensor. |
| **Desviación de Correlación** | Se modifica uno de los dos sensores de un par correlacionado (ej. PRAD y PRGINT) para que se alejen entre sí y rompan la correlación. |
| **Contextual** | Se busca un contexto específico (ej. noche + baja radiación exterior) y se inyecta un valor que solo sería anómalo en ese contexto (ej. radiación interior alta a las 2am). |

Al final de la inyección el dataset tiene **las mismas filas** que el original, pero algunas tienen el valor modificado y su etiqueta cambiada. El modelo aprende a distinguir los patrones anómalos de los normales comparando ambos casos.

> **Importante:** la inyección se hace siempre sobre `df_trabajo` (copia de trabajo), nunca sobre `df_original`. Así podemos comparar original vs modificado durante la evaluación (Fase 7).

---

## Descripción de cada notebook

### `01_carga_datos.ipynb`
- Lee `combined_2024_03_06-2025_11_30_1min.csv`
- Filtra a 2024-03-06 → 2025-03-07 (1 año)
- Añade columnas temporales: `Hora`, `DiaSemana`, `Mes`
- Inicializa etiquetas: `etiqueta_deteccion = 'normal'`, `etiqueta_tipo_anomalia = 'normal'`

### `02_inyeccion_anomalias.ipynb`
Inyecta 6 tipos de anomalías sintéticas:

| Tipo | Descripción |
|------|-------------|
| Datos Faltantes | NaNs en el 2% de filas |
| Sensor Atascado | Valor constante 5-20 minutos |
| Ruido | Picos de 3-5 desviaciones estándar |
| Fuera de Rango | Fuera de percentiles P0.1/P99.9 del dataset |
| Desviación de Correlación | Rompe correlación entre pares de sensores |
| Contextual | Luz nocturna anómala + CO2 sin respuesta a ventilación |

> **Nota v2:** Los límites "Fuera de Rango" son **estadísticos** (percentiles P0.1/P99.9). En v3 se usan los rangos físicos reales del hardware.

### `03_features.ipynb`
- Solo selecciona columnas numéricas relevantes
- **v2 no tiene feature engineering adicional** (sin rolling M1 ni cross-sensor M2)
- El dataset original + columnas temporales son las únicas features

### `04_modelo_deteccion.ipynb`
Entrena el **Modelo 1**: Random Forest binario (normal / anomalía).

> **Diferencia clave con v4:** v2 usa `train_test_split` aleatorio 70/30. v4 usa `TimeSeriesSplit` k=4 para evitar data leakage temporal.

#### train_test_split 70/30 — cómo funciona y sus limitaciones

`train_test_split` baraja todas las filas aleatoriamente y asigna el 70% a train y el 30% a test. Con `random_state=42` el resultado es siempre el mismo (reproducible).

```
Datos mezclados aleatoriamente:
[ene✓] [mar✗] [jun✓] [ago✗] [oct✓] [dic✓] [feb✗] ...
  train   test   train  test   train  train  test
```

**Problema en series temporales:** el modelo ve datos de agosto para predecir enero → **data leakage temporal**. Las métricas quedan infladas y no reflejan el rendimiento real sobre datos futuros.

En v2 este es el baseline de referencia. v4 corrige esto con TimeSeriesSplit k=4, donde siempre el pasado entrena y el futuro se evalúa. La diferencia de métricas entre v2 y v4 cuantifica exactamente el impacto del leakage, lo que resulta útil para el paper.

### `05_modelo_clasificacion.ipynb`
Entrena el **Modelo 2**: Random Forest multiclase (6 tipos de anomalía).
- Solo se entrena con filas que son verdaderas anomalías
- Usa `LabelEncoder` para codificar los 6 tipos

### `06_correccion.ipynb`
Estrategias de corrección por tipo:

| Tipo | Estrategia |
|------|-----------|
| Datos Faltantes | IterativeImputer + RandomForestRegressor |
| Sensor Atascado | Marcar como NaN + IterativeImputer |
| Ruido | Interpolación local (factor umbral = 2.0) |
| Fuera de Rango | Interpolación local (factor umbral = 2.0) |
| Desviación Correlación | Interpolación local (factor umbral = 2.0) |
| Contextual | Interpolación local (factor umbral = 2.0) |

> **Nota v2:** El factor de umbral es **fijo = 2.0** para todos los sensores. En v3 se hace adaptativo por coeficiente de variación (mejora M4).

### `07_evaluacion.ipynb`
- Compara original vs inyectado vs corregido con RMSE y MAE por sensor
- Válido aquí porque tenemos ground truth (anomalías inyectadas artificialmente)

### `08_inferencia_real.ipynb`
- Aplica el pipeline completo a datos reales
- No hay métricas RMSE/MAE válidas (sin ground truth)
- Resultado: listado y visualización de anomalías reales detectadas

---

## Sensores y actuadores (v2 incluye UVENT)

| Columna | Descripción |
|---------|-------------|
| PCO2EXT | CO2 exterior (ppm) |
| PHEXT | Humedad relativa exterior (%) |
| PRAD | Radiación solar exterior (W/m²) |
| PRGINT | Radiación solar interior (W/m²) |
| PTEXT | Temperatura exterior (°C) |
| PVV | Velocidad del viento (m/s) |
| XCO2I | CO2 interior (ppm) |
| XHINV | Humedad relativa interior (%) |
| XTINV | Temperatura interior (°C) |
| XTS | Temperatura suelo (°C) |
| **UVENT_cen** | **Apertura ventanas cenitales (%) — incluido en v2** |
| **UVENT_lN** | **Apertura ventanas laterales (%) — incluido en v2** |

> En v3 se excluyeron UVENT_cen y UVENT_lN porque la señal `_POS/_POS_VALOR` no refleja la posición real del actuador (MAE 40-85% vs encoder físico).

---

## Resultados v2

| Métrica | Valor |
|---------|-------|
| Modelo 1 — Accuracy | 98.85% |
| Modelo 1 — ROC AUC | 0.9946 |
| Modelo 2 — Accuracy | ~90% |
| Sensor Atascado — Recall | 56.3% |
| Fuera de Rango — F1 | 0.81 |
| Anomalías reales detectadas (Fase 8) | 4,811 (0.9% del total) |

---

## Diferencias v2 vs v3

| Aspecto | v2 | v3 |
|---------|----|----|
| Sensores | Incluye UVENT_cen, UVENT_lN | UVENT excluido (señal no fiable) |
| Rangos fuera de rango | Percentiles P0.1/P99.9 | Rangos físicos hardware (M3) |
| Features rolling/lag | No | Sí (M1) |
| Features cross-sensor | No | Sí (M2) |
| Umbrales corrección | Fijos (2.0) | Adaptativos por CV (M4) |
| División train/test | Aleatorio 70/30 | TimeSeriesSplit k=4 (M6) |
| Condición CO2 contextual | Estricta (70%, 50ppm, 5-15min) | Relajada (30ppm, 3-8min) (M9) |
