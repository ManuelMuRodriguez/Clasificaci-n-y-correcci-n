# v4_modular — Pipeline de Detección y Corrección de Anomalías

Versión modular del pipeline de detección, clasificación y corrección de anomalías en datos de invernadero AGROCONNECT.
Cada fase del proceso se separa en un notebook independiente, lo que facilita la comprensión, depuración y mejora incremental.

---

## Estructura del proyecto

```
v4_modular/
├── config.py                      ← Configuración central (rutas, columnas, hiperparámetros)
├── README.md                      ← Este fichero
│
├── 01_carga_datos.ipynb           ← Fase 1: Carga y exploración del dataset
├── 02_inyeccion_anomalias.ipynb   ← Fase 2: Inyección de 6 tipos de anomalías sintéticas
├── 03_features.ipynb              ← Fase 3: Feature engineering (rolling + cross-sensor)
├── 04_modelo_deteccion.ipynb      ← Fase 4: Modelo 1 — detección binaria (normal/anomalía)
├── 05_modelo_clasificacion.ipynb  ← Fase 5: Modelo 2 — clasificación de tipo de anomalía
├── 06_correccion.ipynb            ← Fase 6: Corrección de anomalías (6 estrategias)
├── 07_evaluacion.ipynb            ← Fase 7: Evaluación con ground truth (datos sintéticos)
├── 08_inferencia_real.ipynb       ← Fase 8: Inferencia sobre datos reales del invernadero
│
└── data/
    ├── interim/                   ← DataFrames intermedios entre notebooks (.parquet)
    │   ├── 01_datos_cargados.parquet
    │   ├── 02_datos_inyectados.parquet
    │   ├── 03_datos_features.parquet
    │   ├── 04_modelo1_predicciones.parquet
    │   └── 06_datos_corregidos.parquet
    ├── models/                    ← Modelos entrenados (.joblib)
    │   ├── modelo_1_detector.joblib
    │   ├── imputer_modelo_1.joblib
    │   ├── features_modelo_1.joblib
    │   ├── modelo_2_clasificador.joblib
    │   ├── label_encoder_modelo_2.joblib
    │   └── imputer_datos_faltantes.joblib
    └── plots/                     ← Gráficas generadas por 07 y 08
```

---

## Cómo ejecutar el pipeline

Ejecuta los notebooks en orden numérico. Cada uno carga la salida del anterior desde `data/interim/`.

```
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08
```

Si solo quieres re-ejecutar una parte (por ejemplo, cambiar hiperparámetros del Modelo 1), ejecuta desde el notebook 04 en adelante sin necesidad de repetir la carga de datos ni la inyección.

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
- Lee el CSV combinado (SCADA + OPC UA): `combined_2024_03_06-2025_11_30_1min.csv`
- Filtra al período de análisis: **2024-03-06 → 2025-03-07** (1 año)
- Añade variables temporales: `Hora`, `DiaSemana`, `Mes`
- Verifica NaNs y duplicados
- Inicializa columnas `etiqueta_deteccion` = 'normal' y `etiqueta_tipo_anomalia` = 'normal'
- **Salida:** `data/interim/01_datos_cargados.parquet`

### `02_inyeccion_anomalias.ipynb`
Inyecta anomalías sintéticas para crear el ground truth de entrenamiento. 6 tipos:

| Tipo | Descripción | % aproximado |
|------|-------------|-------------|
| Datos Faltantes | NaNs en sensores aleatorios | 2% de filas |
| Sensor Atascado | Valor constante durante 5-20 minutos | 50 secuencias |
| Ruido / Picos | Desviación de 3-5 σ del valor normal | 3% de filas |
| Valores Fuera de Rango | Fuera de límites físicos del hardware | 2% de filas |
| Desviación de Correlación | Rompe correlación entre sensores pares | 1.5% de filas |
| Contextual | Luz nocturna anómala / CO2 sin ventilación | ~1% + 30 secuencias |

- **Salida:** `data/interim/02_datos_inyectados.parquet`

### `03_features.ipynb`
Añade dos bloques de features derivadas:

**M1 — Features temporales** (por cada sensor):
- `rmean_30m`, `rstd_30m`: media y std en ventana de 30 minutos
- `rmean_3h`, `rstd_3h`: media y std en ventana de 3 horas
- `diff1`, `diff30`: diferencia con punto anterior y con 30 min antes
- `lag1`, `lag30`: valor 1 y 30 minutos antes
- `zscore_local`: desviaciones estándar respecto al rolling_30m

**M2 — Features cruzadas entre sensores**:
- `ratio_rad`: PRGINT / PRAD (ratio radiación interior-exterior)
- `delta_temp`: XTINV - PTEXT (diferencia temperatura dentro-fuera)
- `delta_hum`: XHINV - PHEXT (diferencia humedad dentro-fuera)
- `gradiente_co2`: XCO2I - PCO2EXT (gradiente CO2 interior-exterior)
- `delta_temp_suelo`: XTS - XTINV (diferencia temperatura suelo-aire interior)

- **Salida:** `data/interim/03_datos_features.parquet`

### `04_modelo_deteccion.ipynb`
Entrena el **Modelo 1**: Random Forest para clasificación binaria (normal / anomalía).

Características clave:
- **M6**: TimeSeriesSplit con k=4 — sin data leakage temporal
- SimpleImputer con `add_indicator=True` (los NaN son señal de anomalía)
- Métricas por fold y globales: accuracy, F1, ROC-AUC
- Importancia de features y desglose de detección por tipo

**Salida:** `data/models/modelo_1_detector.joblib`, `imputer_modelo_1.joblib`, `features_modelo_1.joblib`

#### ¿Por qué TimeSeriesSplit y no train_test_split?

`train_test_split` baraja las filas aleatoriamente antes de dividir. Esto produce **data leakage temporal**: el modelo ve datos de agosto para aprender a predecir enero, lo que infla artificialmente las métricas y no refleja el rendimiento real.

Con tus ~553k filas (13 meses), TimeSeriesSplit k=4 divide así:

```
Fold 1: Train [dic 23 – abr 24] → Test [may 24 – jun 24]
Fold 2: Train [dic 23 – jun 24] → Test [jul 24 – ago 24]
Fold 3: Train [dic 23 – ago 24] → Test [sep 24 – oct 24]
Fold 4: Train [dic 23 – oct 24] → Test [nov 24 – dic 24]  ← se guarda
```

Siempre el pasado entrena y el futuro se evalúa — sin leakage.

**Se entrenan 4 modelos independientes**, uno por fold. El objetivo de los folds 1-3 no es elegir el mejor modelo sino **confirmar estabilidad temporal**: ¿funciona igual en primavera que en invierno? Si el accuracy varía mucho entre folds, el modelo no generaliza.

**Siempre se guarda el Fold 4** (no el de mejor accuracy) porque es el que ha entrenado con más datos históricos y ha visto todas las estaciones, siendo el más robusto para producción e inferencia real.

Las métricas que se reportan en el paper son la **media ± desviación** de los 4 folds:

```
Media  Accuracy: 95.5% ± 1.3%
Media  F1:       0.954 ± 0.012
```

Esto es mucho más sólido que un único valor de un solo split, y responde directamente a la pregunta de un revisor: *"¿cómo sabes que generaliza?"*

#### ¿Por qué el Fold 4 tiene menos datos en test que v2?

El test del Fold 4 son solo los ~2 últimos meses (nov-dic 2024), mientras que v2 mezcla el 30% de todo el año:

```
v2 test: [ene✗][mar✗][jun✗][ago✗]... → ~158k filas (30% aleatorio del año completo)
v4 test: [nov 24 → dic 24]           → ~105k filas (solo los 2 últimos meses)
```

v2 tiene ~52k filas más en test, pero incluye datos de todos los meses mezclados — es una ventaja artificial por leakage. A pesar de eso, **v4 obtiene mejor accuracy** (97.87% vs 97.78%), lo que demuestra que el modelo de v4 generaliza mejor de verdad.

#### Pero si el Fold 4 solo evalúa nov-dic, ¿cómo sabemos que funciona en primavera o verano?

Exactamente para eso sirven los folds 1-3. Cada fold evalúa un período distinto del año:

```
Fold 1 → test en may-jun 2024   (primavera)
Fold 2 → test en jul-ago 2024   (verano — máxima radiación)
Fold 3 → test en sep-oct 2024   (otoño)
Fold 4 → test en nov-dic 2024   (invierno) ← modelo guardado
```

Si los 4 folds dan accuracy similar, el modelo funciona bien en todas las estaciones. Si algún fold falla, identifica exactamente en qué período hay problemas. Eso es lo que v2 con un solo 70/30 nunca puede responder.

### `05_modelo_clasificacion.ipynb`
Entrena el **Modelo 2**: Random Forest multiclase para identificar el tipo de anomalía.

- Solo se entrena con filas que son verdaderas anomalías (no con normales)
- Usa LabelEncoder para los 6 tipos de anomalía
- Métricas: accuracy, F1 macro, matriz de confusión

**Salida:** `data/models/modelo_2_clasificador.joblib`, `label_encoder_modelo_2.joblib`

### `06_correccion.ipynb`
Aplica estrategias de corrección específicas por tipo:

| Tipo | Estrategia |
|------|-----------|
| Datos Faltantes | IterativeImputer + RandomForestRegressor (multivariable) |
| Sensor Atascado | Fase híbrida (ver abajo) |
| Ruido | Interpolación local (media de vecinos) |
| Fuera de Rango | Interpolación local con umbrales adaptativos (M4) |
| Desviación Correlación | Interpolación local con umbrales adaptativos (M4) |
| Contextual | Interpolación local con umbrales adaptativos (M4) |

**M4 — Umbrales adaptativos**: el umbral de corrección se ajusta por coeficiente de variación (CV) de cada sensor. Sensores más estables reciben umbrales más estrictos.

**Salida:** `data/interim/06_datos_corregidos.parquet`

#### Fase híbrida — Corrección de Sensor Atascado

Un sensor atascado repite el mismo valor durante varios minutos seguidos. Detectarlo es fácil, pero corregirlo no siempre es igual: depende de si el sensor tiene un patrón temporal fuerte o no.

**El problema con la interpolación lineal en sensores solares:**

PRAD (radiación solar) vale ~800 W/m² al mediodía y 0 W/m² de noche. Si hay un atasco de 3 horas que empieza al mediodía y termina al anochecer, una interpolación lineal daría valores absurdos porque la señal no es lineal — sube y baja con el sol.

```
Atasco PRAD: 12:00 → 15:00, valor congelado en 650 W/m²
Interpolación lineal: 650 → 650 → 650 → ... → valor real 15:00: 400 W/m²
→ Incorrecto: no respeta la curva solar de la tarde
```

Por eso el sensor atascado tiene una **estrategia híbrida** según el tipo de sensor:

---

**Columnas DINÁMICAS** (`PRAD`, `PRGINT`) → corrección estacional

Para cada minuto del atasco, se busca el valor que tenía ese mismo sensor a esa misma hora durante los 7 días anteriores y se toma la mediana:

```
Atasco detectado: PRGINT, 16:27 → 20:43 del 16-nov-2024
↓
Para corregir el minuto 16:27:
  ¿Qué valía PRGINT a las 16:27 el 9-nov? → 180 W/m²
  ¿Qué valía PRGINT a las 16:27 el 10-nov? → 165 W/m²
  ...
  ¿Qué valía PRGINT a las 16:27 el 15-nov? → 172 W/m²
  → mediana = 172 W/m² → valor corregido
Se repite para cada minuto del segmento (16:28, 16:29, ...)
```

---

**Columnas ESTABLES** (XTINV, XHINV, XCO2I, PTEXT, ...) → interpolación lineal

La temperatura, humedad y CO2 cambian lentamente y de forma suave. Una línea recta entre el valor antes y después del atasco es suficientemente precisa:

```
Atasco detectado: XTINV, valor 22.3°C repetido durante 8 minutos
↓
Valor justo antes del atasco:  22.1°C
Valor justo después del atasco: 22.6°C
→ Interpolación lineal: 22.2, 22.3, 22.4, 22.5, 22.6
→ Correcto: la temperatura sube gradualmente
```

---

**Flujo completo de la fase híbrida:**

```
Modelo 2 predice "Sensor Atascado" en un segmento
              ↓
   ¿Duración ≥ 5 minutos continuos?
              ↓ sí
   ┌──────────────────────────────┐
   │ ¿Es PRAD o PRGINT?          │
   │  Sí → corrección estacional  │  (misma hora, mediana 7 días atrás)
   │  No → interpolación lineal   │  (línea recta entre extremos)
   └──────────────────────────────┘
```

En la ejecución real sobre los datos de nov-dic 2024:
- **PRAD**: 39 segmentos atascados → 25.659 minutos corregidos estacionalmente
- **PRGINT**: 195 segmentos atascados → 27.945 minutos corregidos estacionalmente

### `07_evaluacion.ipynb`
Evaluación cuantitativa **solo válida para datos sintéticos** (hay ground truth):
- RMSE y MAE por sensor: original vs corregido
- Gráficas comparativas de los tres estados (original / inyectado / corregido)
- Tabla de métricas por sensor

### `08_inferencia_real.ipynb`
Aplica el pipeline completo a datos reales del invernadero:
- Carga el dataset combinado sin anomalías sintéticas
- Aplica feature engineering (mismas transformaciones que en entrenamiento)
- Modelo 1: detecta anomalías reales
- Modelo 2: clasifica el tipo
- Modelos de corrección: corrige cada anomalía
- Desglose final por tipo de anomalía real detectada

> **Importante:** No existen métricas RMSE/MAE válidas en este notebook. La validación es cualitativa mediante inspección visual por experto de dominio.

---

## Sensores y actuadores

| Columna | Descripción | Rango físico |
|---------|-------------|-------------|
| PCO2EXT | CO2 exterior (ppm) | 0 – 2000 |
| PHEXT | Humedad relativa exterior (%) | 0 – 100 |
| PRAD | Radiación solar exterior (W/m²) | 0 – 1500 |
| PRGINT | Radiación solar interior (W/m²) | 0 – 1200 |
| PTEXT | Temperatura exterior (°C) | -20 – 60 |
| PVV | Velocidad del viento (m/s) | 0 – 50 |
| XCO2I | CO2 interior (ppm) | 300 – 3000 |
| XHINV | Humedad relativa interior (%) | 0 – 100 |
| XTINV | Temperatura interior (°C) | -5 – 60 |
| XTS | Temperatura suelo (°C) | -5 – 70 |

> **Excluidos del modelo:** `UVENT_cen` y `UVENT_lN` (apertura ventanas cenitales y laterales).
> Motivo: la señal `_POS/_POS_VALOR` no refleja la posición real del actuador (MAE 40-85% vs encoder físico).
> Su efecto queda recogido implícitamente en `XTINV`, `XHINV` y `XCO2I`.

---

## Comparativa v2 → v3 → v4

| Mejora | v2 | v3 | v4 (pendiente) |
|--------|----|----|----------------|
| Features rolling/lag | No | Sí (M1) | Sí |
| Features cruzadas | No | Sí (M2) | Sí |
| Rangos físicos hardware | No (percentiles) | Sí (M3) | Sí |
| Umbrales adaptativos CV | No | Sí (M4) | Sí |
| TimeSeriesSplit | No (random 70/30) | Sí k=4 (M6) | Sí |
| UVENT excluido | No | Sí (M7) | Sí |
| Condición CO2 relajada | No | Sí (M9) | Sí |
| Modelo único multiclase | No | No | Pendiente |
| LightGBM vs RF | No | No | Pendiente |
| `ensayo` como feature | No | No | Pendiente |
| Notebooks modulares | No | No | **Sí (este repo)** |

---

## Resultados v2 (referencia)

| Métrica | Valor |
|---------|-------|
| Modelo 1 — Accuracy | 98.85% |
| Modelo 1 — ROC AUC | 0.9946 |
| Modelo 2 — Accuracy | ~90% |
| Sensor Atascado — Recall | 56.3% (KPI a mejorar) |
| Fuera de Rango — F1 | 0.81 (KPI a mejorar) |
| Anomalías reales detectadas (Fase 7) | 4,811 (0.9% del total) |

---

## Trabajo futuro (v5)

- **Modelo único multiclase**: eliminar la cascada M1 → M2, reducir errores propagados
- **LightGBM**: x5-10 más rápido que RF, especialmente útil en corrección
- **`ensayo` como feature one-hot**: los días de experimento tienen patrones distintos
- **Validación 2025-2026**: re-aplicar Fase 8 sobre el segundo año de datos cuando esté disponible
- **SHAP values**: explicabilidad de predicciones para validación con experto
