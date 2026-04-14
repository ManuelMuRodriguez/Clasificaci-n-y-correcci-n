# Dudas y Respuestas — Análisis Técnico del Proyecto

> **Nota:** Todo lo documentado aquí se basa en el notebook **"Ultimo antes del cambio Clasificación_y_Corrección_de_Anomalías.ipynb"**, que es el notebook activo del proyecto. El fichero "Versión TFM..." es una versión anterior y no se usa como referencia.

---

## 1. ¿El modelo aprende o son reglas?

**Es un híbrido de los dos**, y en cada etapa se usa una cosa diferente:

| Etapa | Mecanismo |
|---|---|
| **Inyección** de anomalías | Reglas basadas en percentiles del propio dataset |
| **Detección** Normal vs Anomalía | ML — Random Forest aprende de los patrones |
| **Clasificación** del tipo | ML — Random Forest aprende a distinguir los 6 tipos |
| **Corrección** | Reglas (vecinos locales, interpolación, imputer) |

El Random Forest **aprende** de los datos con anomalías sintéticas. No aplica reglas explícitas tipo `if valor > max: anomalía`.

---

## 2. ¿Se usan features temporales?

Hay **tres features temporales** — pero son muy básicas:

```python
# Lo que extrae del campo Fecha:
Hora        # hora del día (0-23)
DiaSemana   # día de la semana (0-6)
Mes         # mes del año (1-12)
```

**Lo que NO hay:**
- `rolling_std(window=N)` — desviación estándar móvil
- `diff` — diferencia con el instante anterior
- `lag_1, lag_2...` — valores anteriores
- Nada que capture la evolución en el tiempo

El modelo ve cada fila de forma **casi independiente**. La única "memoria" temporal son esas 3 columnas de calendario.

---

## 3. Fuera de rango: ¿límite físico del sensor o estadístico?

**Corrección respecto a versiones anteriores de este documento:** los rangos en el notebook activo son **estadísticos, calculados automáticamente del propio dataset**, no rangos físicos hardcodeados.

```python
# Se calculan los percentiles 0.1 y 99.9 de cada sensor sobre los datos originales
limites_automaticos[col] = {
    'min': p0_1,    # percentil 0.1 del dataset
    'max': p99_9    # percentil 99.9 del dataset
}
rangos_validos_sensores = limites_automaticos
```

La inyección introduce valores a un **10% más allá de esos límites estadísticos**:

```python
margen_factor = 0.1  # 10% del span del rango

# Si para PTEXT el P0.1 = 3°C y P99.9 = 28°C (span = 25°C):
# valor anómalo alto  = 28 + (25 × 0.1) = 30.5°C
# valor anómalo bajo  =  3 - (25 × 0.1) = 0.5°C
```

### Rangos físicos reales — extraídos de las fichas técnicas (PLCs)

Con los metadatos reales del sistema AGROCONNECT (fichero `Dataset/Metadatos_SensoresyActuadores/AGROCONNECT_Variables_PLCs.xlsx`) se dispone de los rangos físicos certificados de cada sensor. Estos son los límites correctos para la categoría "Valores Fuera de Rango" — cualquier valor fuera de ellos es **físicamente imposible** para ese hardware:

| Variable paper | Columna real | Sensor / Modelo | Rango físico real | Precisión |
|---|---|---|---|---|
| `PCO2EXT` | `CO2_EXTERIOR_10M` | E+E Elektronik EE820-HV1A6E1 | **0 – 2000 ppm** | ±(50 ppm + 2% del valor) |
| `PHEXT` | `HR_EXTERIOR_10M` | Campbell HC2A-S3 | **0 – 100 %** | ±0.8 % rH |
| `PRAD` | `RADGLOBAL_EXTERIOR_10M` | Campbell SP-214-SS | **0 – 2000 W/m²** | — |
| `PRGINT` | `INVER_RADGLOBAL_INTERIOR_S1` | Campbell SP-110-SS | **0 – 2000 W/m²** | ±1 % |
| `PTEXT` | `TEMP_EXTERIOR_10M` | Campbell HC2A-S3 | **-40 – 60 °C** | ±0.1 °C |
| `PVV` | `VELVIENTO_EXTERIOR_10M` | Wittich & Visser PA2 | **0 – 45.8 m/s** (165 km/h) | 0.5 m/s |
| `UVENT_cen` | Media UVCEN*_POS | De Gier I-DE (encoder) | **0 – 100 %** | lineal 4–20 mA |
| `UVENT_lN` | Media UVLAT*_POS | De Gier I-DE (encoder) | **0 – 100 %** | lineal 4–20 mA |
| `XCO2I` | `INVER_CO2_INTERIOR_S1` | E+E Elektronik EE820-HV1A6E1 | **0 – 2000 ppm** | ±(50 ppm + 2% del valor) |
| `XHINV` | `INVER_HR_INTERIOR_S1` | Campbell HC2A-S3 | **0 – 100 %** | ±0.8 % rH |
| `XTINV` | `INVER_TEMP_INTERIOR_S1` | Campbell HC2A-S3 | **-40 – 60 °C** | ±0.1 °C |
| `XTS` | `INVER_TEMP_SUELO5_S1` | Campbell 109 (termistor NTC) | **-50 – 70 °C** | ±0.2 °C |

**Fuente:** PLC1 (sensores interiores S1), PLC6 - Mástil Exterior. Fichero: `AGROCONNECT_Variables_PLCs.xlsx`

Estos rangos deben reemplazar los percentiles P0.1/P99.9 como definición de "Valores Fuera de Rango" en el sistema. Los percentiles estadísticos siguen siendo útiles pero para otra categoría: valores inusuales en el contexto estacional (ver Sección 4).

---

## 4. El problema de la estacionalidad con percentiles globales

Usar P0.1/P99.9 del **dataset completo** es mejor que rangos físicos hardcodeados, pero **sigue siendo un enfoque global que no captura la estacionalidad**.

El problema concreto:

- El dataset cubre octubre a diciembre 2020. Si en ese período la temperatura exterior máxima fue 28°C, el P99.9 de PTEXT se fijará en ≈28°C.
- Un valor de `PTEXT = 15°C` en octubre es perfectamente normal, pero `PTEXT = 15°C` en diciembre a las 14:00h podría ser anómalo para esa franja.
- El percentil global no distingue entre épocas: define lo que es "raro en el conjunto total" pero no "raro para esta hora y este mes".

**Lo que ocurre en la práctica:** el modelo detecta como "Fuera de Rango" solo lo que supera el percentil extremo del dataset completo. Anomalías que son inusuales solo para una época o franja horaria concreta quedan sin detectar, etiquetadas como "normal".

---

## 5. Fallo de diseño de fondo en la categoría "Fuera de Rango"

Aunque el enfoque de percentiles globales es más robusto que rangos físicos, **sigue sin capturar la anormalidad contextual**.

Lo que el ML añadiría valor de verdad es detectar valores **dentro del rango global pero fuera del rango contextual** (anómalo para esa hora, ese mes, ese estado del invernadero). Para eso se necesitaría:

**Rangos adaptativos por contexto estacional:**
```python
# En lugar de P99.9 global:
rangos_por_contexto = df.groupby(['Mes', 'Hora'])['PTEXT'].agg(
    p01=lambda x: x.quantile(0.01),
    p99=lambda x: x.quantile(0.99)
)
# Anomalía = valor fuera del percentil 1-99 para ese mes y hora concretos
```

**Features estadísticas por contexto:**
```python
# z-score del valor respecto a la media de esa hora + mes
df['PTEXT_zscore_contexto'] = df.groupby(['Hora', 'Mes'])['PTEXT'].transform(
    lambda x: (x - x.mean()) / x.std()
)
```

**Features de ventana temporal (rolling):**
```python
df['PTEXT_rolling_std'] = df['PTEXT'].rolling(window=6).std()   # últimos 30 min
df['PTEXT_diff']        = df['PTEXT'].diff()                     # cambio respecto al instante anterior
```

---

## 6. ¿Existen features de relación entre sensores?

**No en el modelo.** Es importante distinguir dos cosas que ocurren en el proyecto:

### La matriz de correlación existe — pero solo se usa para inyectar anomalías

El código calcula la matriz de correlación del dataset original:

```python
correlation_matrix = df_original[columnas_numericas_para_corr].corr()
```

Y la usa para crear anomalías de "Desviación de Correlación" realistas:

```python
pares_para_desviacion = [
    ('PRAD', 'PRGINT'),    # Correlación muy fuerte positiva
    ('XTINV', 'XHINV'),   # Correlación fuerte negativa
    ('PRAD', 'UVENT_cen') # Correlación fuerte positiva (operacional)
]
# Se modifica un sensor del par para que viole la correlación esperada con el otro
corr_val = correlation_matrix.loc[sensor_a_modificar, sensor_referencia]
```

Es decir: **el sistema sabe que PRAD y PRGINT están correlacionados**, lo usa para generar anomalías realistas, pero **nunca le da esa información al modelo**. Es la paradoja central del diseño: usa las correlaciones para crear el problema pero luego oculta esas correlaciones al modelo que tiene que resolverlo.

### Lo que entra al modelo (X) — verificado en el código

```python
columnas_potenciales_features = [col for col in df_trabajo.columns
                                  if col not in ['Fecha', 'etiqueta_deteccion', 'etiqueta_tipo_anomalia']]
X = df_trabajo[columnas_potenciales_features].copy()
```

Exactamente esto:
```
PCO2EXT | PHEXT | PRAD | PRGINT | PTEXT | PVV | XCO2I | XHINV | XTINV | XTS | UVENT_cen | UVENT_lN | Hora | DiaSemana | Mes | [indicadores NaN]
```

Cada columna es un valor puntual en ese instante. No existe ninguna feature de tipo:
- `PRGINT / PRAD` (ratio de transmisión del cristal)
- `XTINV - PTEXT` (diferencia temperatura interior-exterior)
- `XCO2I × UVENT_cen` (respuesta del CO2 a la ventilación)

El modelo puede aprender que "PTEXT=25 en Mes=12 es raro", pero **no puede aprender que "XTINV=44 cuando PTEXT=10 es imposible"** porque esa relación nunca se le presenta como feature.

---

## 7. Relaciones entre sensores que el modelo no puede ver

Estas son las correlaciones más relevantes del sistema invernadero que actualmente son **invisibles para el modelo**:

### Temperatura exterior ↔ Temperatura interior
La relación no es fija — depende de la radiación solar y la ventilación:

| Situación | PTEXT | XTINV | Relación |
|---|---|---|---|
| Invierno, sin sol, ventanas cerradas | 10°C | 8°C | Interior ligeramente menor |
| Invierno, con sol, ventanas cerradas | 10°C | 22°C | Sol calienta el interior |
| Verano, sin ventilación | 38°C | 44°C | Efecto invernadero |
| Verano, ventanas abiertas | 38°C | 39°C | Temperaturas igualadas |

Un valor de `XTINV=44°C` con `PTEXT=10°C` y `UVENT_cen=0` es una **anomalía clara** aunque los tres valores individualmente estén dentro de su rango estadístico. El modelo actual no puede detectarlo.

### Radiación exterior ↔ Radiación interior
La relación es casi lineal con un factor de transmisión del cristal:

```
PRGINT ≈ PRAD × factor_transmisión  (factor típico: 0.4–0.7)
```

- De noche `PRAD=0` → `PRGINT` debería ser 0. Si es > 0: anomalía (ya implementada como contextual).
- De día `PRAD=800` → `PRGINT` debería estar entre 320 y 560. Si es 0: sensor interior roto.
- Si el ratio cambia bruscamente de un día para otro: posible suciedad en el cristal o fallo de sensor.

El modelo ve ambas columnas por separado. **No calcula el ratio ni detecta cuando se rompe la proporción esperada.**

### Ventilación ↔ Temperatura interior + Humedad interior + CO2 interior
Los actuadores de ventilación (`UVENT_cen`, `UVENT_lN`) causan cambios en las variables interiores:

| Ventilación | Efecto esperado en interior |
|---|---|
| UVENT abre → | XTINV se acerca a PTEXT |
| UVENT abre → | XHINV se acerca a PHEXT |
| UVENT abre → | XCO2I baja hacia PCO2EXT |
| UVENT cierra → | Divergencia gradual entre interior y exterior |

**Lo que el modelo no puede ver:** si `UVENT_cen=100` y `XTINV` no se mueve durante 30 minutos con diferencia de temperatura alta, eso es una anomalía. Pero requiere el **efecto acumulado en el tiempo**, no el valor puntual.

### Ventilación ↔ Variable tiempo (inercia del sistema)
Los actuadores no tienen efecto instantáneo:

- Cuando se abre la ventilación, la temperatura no cae en 5 minutos, sino en 20–40 minutos.
- El suelo (`XTS`) responde con horas de retraso a cambios de temperatura interior.
- Un sensor que "no responde" a un actuador solo es detectable mirando la evolución temporal.

El modelo ve el instante `t` de forma aislada. **No tiene acceso a lo que pasó en `t-1, t-2, ..., t-N`**.

### Temperatura suelo ↔ Temperatura interior (inercia térmica)
`XTS` debería seguir a `XTINV` con retraso y suavidad:

- Cambio brusco en `XTS` en 5 minutos → físicamente imposible → anomalía.
- `XTS` constante durante horas mientras `XTINV` cambia mucho → sensor atascado.

Ambos casos requieren comparar el valor actual con valores anteriores. Sin features de serie temporal, el modelo los ve como valores puntuales normales.

---

## 8. Resumen de relaciones ausentes

| Relación | Variables | Por qué es importante | Ausente en el modelo |
|---|---|---|---|
| Temperatura exterior ↔ interior | PTEXT → XTINV | Correlación condicional a radiación y ventilación | Sí |
| Radiación exterior ↔ interior | PRAD → PRGINT | Ratio casi constante; ruptura = anomalía | Sí |
| Ventilación → temperatura interior | UVENT → XTINV | Causalidad con retraso temporal | Sí |
| Ventilación → humedad interior | UVENT → XHINV | Causalidad con retraso temporal | Sí |
| Ventilación → CO2 interior | UVENT → XCO2I | Causalidad con retraso temporal | Parcialmente (intento fallido) |
| Inercia del suelo | XTINV → XTS | XTS varía lento; cambio brusco = anomalía | Sí |
| Comportamiento estacional de cada par | (Mes, Hora) → relación normal | Rango de relación cambia según época | Sí |

**Consecuencia directa:** el modelo detecta bien las anomalías más obvias (NaN, fuera del percentil global, spikes muy grandes) pero es ciego a anomalías sutiles que solo se manifiestan cuando se rompe la relación esperada entre variables o cuando un sensor no responde como debería a lo que hace el sistema.

---

## 9. Otros aspectos del sistema — verificados en el notebook activo

### Split train/test
El split es **aleatorio estratificado**, no temporal:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,       # 70% train, 30% test
    random_state=42,
    stratify=y           # Mantiene proporción de clases
)
```
El 30% de test puede contener datos anteriores al 70% de train. En series temporales esto es data leakage — el modelo "ve el futuro" durante el entrenamiento.

### Arquitectura de los modelos
Ambos modelos son `RandomForestClassifier` con exactamente los mismos hiperparámetros:
```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced',  # Compensa el desbalance normal/anomalía
    n_jobs=-1
)
```
No hay búsqueda de hiperparámetros (GridSearch, RandomSearch). Los valores son fijos y sin justificación documentada.

### Imputer de datos faltantes (corrección)
La corrección de datos faltantes usa `IterativeImputer` con un Random Forest interno:
```python
IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=30, max_depth=10, min_samples_leaf=5),
    max_iter=10,
    initial_strategy='median'
)
```

### Threshold de corrección — igual para todos los sensores
Las cuatro funciones de corrección (Ruido, Fuera de Rango, Desviación de Correlación, Contextual) usan el mismo factor:
```python
factor_umbral_diferencia = 2.0  # Igual en los 4 métodos de corrección
threshold = 2.0 × std_dev_del_sensor
```
No es adaptativo por sensor. Un sensor muy estable (XTS) y uno muy variable (PRAD) reciben el mismo factor.

### Modelos guardados con joblib
A diferencia de versiones anteriores, **en el notebook activo los modelos sí se guardan**:
```python
joblib.dump(imputer,               'imputador_nans.joblib')
joblib.dump(rf_detector,           'modelo_1_detector.joblib')
joblib.dump(columnas_potenciales_features, 'features_modelo_1.joblib')
joblib.dump(rf_clasificador_tipo,  'modelo_2_clasificador.joblib')
joblib.dump(label_encoder_model2,  'label_encoder_modelo_2.joblib')
```

### CO2 contextual — 0 secuencias generadas
El código intenta inyectar 100 secuencias de "CO2 sin respuesta a ventilación" pero no encuentra ventanas válidas en el dataset (ventilación alta + gradiente CO2 suficiente + segmento normal). El resultado es 0 secuencias. Solo la anomalía contextual de "luz nocturna" funciona correctamente (538 filas).

---

## Resumen general

| Aspecto | Lo que se creyó | Lo que realmente hace | Verificado |
|---|---|---|---|
| Features temporales de serie | Sí usaba | Solo Hora, DiaSemana, Mes — sin rolling, diff ni lags | ✓ Confirmado |
| Correlaciones entre sensores como features | Sí las usaba | Matriz de correlación existe pero solo para inyectar anomalías, nunca como feature del modelo | ✓ Confirmado |
| Actuadores ↔ sensores como features | Sí los relacionaba | UVENT entra como columna bruta, sin features de relación con sensores interiores | ✓ Confirmado |
| "Fuera de rango" — cómo se define | Rangos físicos hardcodeados | Percentiles P0.1/P99.9 calculados del dataset (estadístico global, no contextual) | ✓ Corregido |
| Anomalías estacionales | Detectadas | No — el percentil global no distingue si un valor es anómalo para una época concreta | ✓ Confirmado |
| Split train/test | Con cuidado temporal | Aleatorio estratificado — riesgo de data leakage en series temporales | ✓ Confirmado |
| Hiperparámetros del modelo | Optimizados | Fijos sin búsqueda (n_estimators=100 sin justificación empírica) | ✓ Confirmado |
| Modelos guardados a disco | No | Sí — joblib.dump para los dos RF y el encoder | ✓ Corregido |
| Threshold de corrección | Adaptado por sensor | 2×std fijo igual para todos los sensores | ✓ Confirmado |
| CO2 contextual | Implementado y funcionando | Código existe pero genera 0 secuencias por falta de ventanas válidas | ✓ Confirmado |

---

---

# Propuestas de Mejora para Publicación Científica

Las mejoras se organizan en cinco bloques: features, modelo, validación, corrección y arquitectura general. Para cada propuesta se indica la debilidad que resuelve y la referencia en literatura cuando aplica.

---

## M1 — Features de Serie Temporal (Temporalidad)

**Debilidad que resuelve:** el modelo ve cada instante de forma aislada, sin acceso a lo que ocurrió antes. Esto impide detectar sensor atascado (valor constante durante N intervalos), spikes transitorios, y la falta de respuesta de un sensor a un actuador.

### Ventana deslizante (rolling)

```python
window_corta = 6    # 30 minutos (6 × 5 min)
window_larga = 36   # 3 horas

for col in sensores:
    df[f'{col}_rmean_30m']  = df[col].rolling(window_corta).mean()
    df[f'{col}_rstd_30m']   = df[col].rolling(window_corta).std()
    df[f'{col}_rmean_3h']   = df[col].rolling(window_larga).mean()
    df[f'{col}_rstd_3h']    = df[col].rolling(window_larga).std()
```

- `rstd_30m ≈ 0` durante muchos intervalos → **Sensor Atascado** (resuelve el 82.4% de detección actual)
- `rstd_30m` muy alto respecto a `rstd_3h` → **Ruido/Spike**

### Diferencias y lags

```python
for col in sensores:
    df[f'{col}_diff1']  = df[col].diff(1)   # cambio en 5 min
    df[f'{col}_diff6']  = df[col].diff(6)   # cambio en 30 min
    df[f'{col}_lag1']   = df[col].shift(1)  # valor hace 5 min
    df[f'{col}_lag6']   = df[col].shift(6)  # valor hace 30 min
```

`diff1` muy grande → spike. `diff1 = 0` repetido → atascado. Los lags permiten al modelo comparar el estado actual con el reciente.

### z-score local (posición respecto a la tendencia reciente)

```python
df[f'{col}_zscore_local'] = (
    (df[col] - df[f'{col}_rmean_30m']) /
    (df[f'{col}_rstd_30m'] + 1e-6)
)
```

`|zscore_local| > 3` → anomalía local (spike o fuera de rango respecto a su propio contexto).

**Referencia:** Chandola, V., Banerjee, A., Kumar, V. (2009). *Anomaly Detection: A Survey*. ACM Computing Surveys.

---

## M2 — Features de Relación entre Variables (Correlaciones)

**Debilidad que resuelve:** el modelo no puede detectar que la relación entre dos variables se ha roto, aunque los valores individuales sean normales.

### Ratio radiación interior / exterior

```python
df['ratio_rad'] = df['PRGINT'] / (df['PRAD'] + 1)
# Normal: 0.3–0.7 (factor de transmisión del cristal)
# Anómalo: ratio ≈ 0 con PRAD alto → sensor PRGINT roto
# Anómalo: ratio >> 1 de noche → luz artificial o fallo
```

### Diferencia temperatura interior - exterior

```python
df['delta_temp'] = df['XTINV'] - df['PTEXT']
# Invierno sin sol: delta ≈ -2 a +5
# Verano con sol cerrado: delta ≈ +4 a +10
# Anómalo: delta muy alto en invierno sin radiación
# Anómalo: delta muy bajo en verano con ventanas cerradas
```

### Diferencia humedad interior - exterior

```python
df['delta_hum'] = df['XHINV'] - df['PHEXT']
# Con ventilación abierta: delta → 0
# Con ventilación cerrada: puede diferir
# Anómalo: delta muy grande con ventilación máxima abierta
```

### Efecto de la ventilación sobre el CO2

```python
df['vent_total'] = df['UVENT_cen'] + df['UVENT_lN']
df['gradiente_co2'] = df['XCO2I'] - df['PCO2EXT']
df['respuesta_co2_vent'] = df['gradiente_co2'] * df['vent_total']
# Si ventilación alta y gradiente alto → XCO2I debería bajar
# Si este producto es alto durante mucho tiempo → fallo del sistema de ventilación
```

### Inercia del suelo respecto al interior

```python
df['delta_temp_suelo'] = df['XTS'] - df['XTINV']
df['XTS_diff1'] = df['XTS'].diff(1)
# XTS_diff1 grande en 5 minutos → físicamente imposible → anomalía
# XTS constante mientras XTINV varía mucho → sensor atascado
```

**Referencia:** Martineau, M. et al. (2018). *A Survey of Machine Learning Methods for IoT Sensor Data Anomaly Detection*. IEEE Access.

---

## M3 — Rangos Contextuales por Estación y Franja Horaria

**Debilidad que resuelve:** los percentiles P0.1/P99.9 son globales. Un valor normal en verano puede ser anómalo en invierno y viceversa. El modelo no puede distinguirlos.

### Percentiles por grupo (Mes × Hora)

```python
# Calcular con datos normales (antes de inyectar anomalías)
stats_contextuales = df_normal.groupby(['Mes', 'Hora'])[sensores].agg(
    ['mean', 'std',
     lambda x: x.quantile(0.01),
     lambda x: x.quantile(0.99)]
)

# Feature: posición del valor dentro del rango normal de ese contexto
for col in sensores:
    p01 = stats_por_contexto[(col, 'p01')]  # percentil 1 para ese Mes+Hora
    p99 = stats_por_contexto[(col, 'p99')]  # percentil 99 para ese Mes+Hora
    df[f'{col}_pos_contextual'] = (df[col] - p01) / (p99 - p01 + 1e-6)
    # 0–1: rango normal contextual
    # > 1 o < 0: anómalo para esa época y franja horaria
```

Este enfoque resuelve directamente el ejemplo planteado: `PTEXT=25°C` en diciembre a las 14h daría `pos_contextual >> 1` aunque globalmente sea un valor posible.

**Referencia:** Ding, N. et al. (2021). *Time-series Anomaly Detection for Smart Agriculture*. Sensors.

---

## M4 — Corrección: Thresholds Adaptativos por Sensor

**Debilidad que resuelve:** el factor `2×std` fijo es demasiado permisivo para sensores estables (XTS, PCO2EXT) y demasiado agresivo para sensores muy variables (PRAD, PVV).

```python
# Calcular el coeficiente de variación de cada sensor
cv_sensor = {
    col: df[col].std() / (df[col].mean() + 1e-6)
    for col in sensores
}

# Factor adaptativo: más exigente con sensores estables
def factor_por_sensor(col):
    cv = cv_sensor[col]
    if cv < 0.1:   return 1.5   # sensor muy estable → threshold más estricto
    elif cv < 0.5: return 2.0   # variabilidad moderada
    else:          return 2.5   # sensor muy variable → threshold más permisivo

threshold = factor_por_sensor(col) * std_dev_sensor
```

---

## M5 — Corrección: Métodos más Robustos por Tipo

**Debilidad que resuelve:** el método de vecinos inmediatos con promedio simple no captura bien las dinámicas de sensores con alta variabilidad legítima (PRAD, PRGINT).

### Filtro de Kalman para corrección suave

Para sensores con dinámica conocida y suave (XTS, XTINV), el filtro de Kalman estima el valor verdadero a partir de medidas ruidosas usando un modelo de transición:

```
estado_t = F × estado_(t-1) + ruido_proceso
medida_t = H × estado_t + ruido_medida
```

Produce correcciones suaves y respeta la inercia física del sensor.

**Referencia:** Welch, G., Bishop, G. (2006). *An Introduction to the Kalman Filter*. UNC Chapel Hill.

### Imputación basada en correlación física

Para spikes en PRGINT cuando PRAD es válido:

```python
# Usar la relación conocida PRGINT ≈ PRAD × factor
# en lugar de promediar vecinos
factor_transmision = df_normal['PRGINT'].mean() / df_normal['PRAD'].mean()
valor_corregido = df.loc[idx, 'PRAD'] * factor_transmision
```

### Corrección condicional a la confianza del modelo

Solo corregir cuando el modelo tiene alta confianza, para evitar correcciones erróneas:

```python
prob_anomalia = rf_detector.predict_proba(X)[:, 1]
umbral_confianza = 0.85

# Solo corregir si el modelo está seguro
mask_corregir = (prediccion == 'anomalia') & (prob_anomalia > umbral_confianza)
```

---

## M6 — Validación Temporal Correcta

**Debilidad que resuelve:** el split aleatorio permite data leakage — el modelo ve datos "futuros" durante el entrenamiento, inflando artificialmente las métricas.

### TimeSeriesSplit

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    # El test siempre es temporalmente posterior al train
```

### Walk-forward validation

Más estricto: entrenar con los primeros N días y evaluar en el siguiente bloque de M días, avanzando en el tiempo:

```
Fold 1: train [Oct 10–Nov 10] → test [Nov 11–Nov 20]
Fold 2: train [Oct 10–Nov 20] → test [Nov 21–Nov 30]
Fold 3: train [Oct 10–Nov 30] → test [Dic 1–Dic 10]
...
```

**Referencia:** Bergmeir, C., Benítez, J.M. (2012). *On the use of cross-validation for time series predictor evaluation*. Information Sciences.

---

## M7 — Modelos Alternativos y Complementarios

El Random Forest supervisado requiere anomalías etiquetadas para entrenarse, lo que obliga a la inyección sintética. Los modelos de detección de anomalías en literatura también incluyen enfoques no supervisados que aprenden la distribución normal sin necesitar etiquetas:

### Isolation Forest (no supervisado)

Detecta anomalías aislándolas en árboles de decisión aleatorios. Cuanto más fácil es aislar un punto, más anómalo es. No necesita etiquetas.

```python
from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination=0.085, random_state=42)
iso.fit(X_normal)
scores = iso.decision_function(X_nuevo)
```

**Ventaja:** no depende de anomalías sintéticas, aprende directamente de datos normales.
**Referencia:** Liu, F.T., Ting, K.M., Zhou, Z.H. (2008). *Isolation Forest*. ICDM.

### LSTM Autoencoder (deep learning temporal)

Aprende a reconstruir secuencias temporales normales. Una secuencia que no puede reconstruir bien es anómala. Captura naturalmente las dependencias temporales y las correlaciones entre sensores.

```
Entrada: ventana de T instantes × N sensores
Encoder LSTM → representación comprimida
Decoder LSTM → reconstrucción de la ventana
Error de reconstrucción alto → anomalía
```

**Ventaja:** modela temporalidad y correlaciones sin feature engineering explícito.
**Referencia:** Malhotra, P. et al. (2016). *LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection*. ICML Workshop.

### One-Class SVM

Aprende la frontera del espacio de datos normales. Cualquier punto fuera de esa frontera es anómalo.

**Ventaja:** funciona bien con datos de alta dimensionalidad y no requiere etiquetas de anomalía.
**Referencia:** Schölkopf, B. et al. (2001). *Estimating the Support of a High-Dimensional Distribution*. Neural Computation.

### Matrix Profile / STUMPY (anomalías en subsecuencias)

Detecta subsecuencias de tiempo que no tienen ninguna subsecuencia similar en la serie (discords). Muy efectivo para sensor atascado y spikes.

```python
import stumpy
mp = stumpy.stump(df['XTINV'].values, m=12)  # ventana de 12 intervalos (1 hora)
# mp[:, 0] contiene el perfil de distancia — valores altos = discord/anomalía
```

**Referencia:** Yeh, C.C.M. et al. (2016). *Matrix Profile I: All Pairs Similarity Joins for Time Series*. ICDM.

### Descomposición STL + detección de residuos

Separa la serie en tendencia + estacionalidad + residuo. Las anomalías aparecen como residuos grandes:

```python
from statsmodels.tsa.seasonal import STL
stl = STL(df['XTINV'], period=288)  # 288 intervalos de 5 min = 1 día
result = stl.fit()
residuo = result.resid
anomalias = np.abs(residuo) > 3 * residuo.std()
```

**Ventaja:** elimina la variabilidad estacional antes de buscar anomalías, resolviendo el problema de estacionalidad directamente.
**Referencia:** Cleveland, R.B. et al. (1990). *STL: A Seasonal-Trend Decomposition Procedure Based on Loess*. Journal of Official Statistics.

---

## M8 — Búsqueda de Hiperparámetros

**Debilidad que resuelve:** los `n_estimators=100` actuales no tienen justificación empírica. Pueden ser insuficientes o excesivos.

```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.3]
}

search = RandomizedSearchCV(
    RandomForestClassifier(class_weight='balanced', n_jobs=-1),
    param_distributions=param_dist,
    n_iter=30,
    cv=TimeSeriesSplit(n_splits=5),  # combinado con M6
    scoring='f1_weighted',
    random_state=42
)
```

---

## M9 — Anomalía CO2 Contextual: Relajar Condiciones de Inyección

**Debilidad que resuelve:** el escenario "CO2 sin respuesta a ventilación" genera 0 secuencias porque las condiciones son demasiado restrictivas.

Opciones para generar secuencias válidas:

```python
# Condición actual (muy restrictiva):
# UVENT_cen > 70% AND UVENT_lN > 70% AND gradiente_CO2 > 50 ppm

# Condición relajada:
umbral_vent = 40          # en lugar de 70
umbral_gradiente_co2 = 30 # en lugar de 50 ppm
duracion_minima = 3       # en lugar de buscar 5 intervalos consecutivos
```

Alternativamente, generar el escenario de forma completamente sintética sobre un segmento de datos elegido manualmente donde las condiciones se cumplan parcialmente.

---

## Resumen de mejoras priorizadas

| Prioridad | Mejora | Problema que resuelve | Complejidad |
|---|---|---|---|
| Alta | M1 — Rolling + diff + lag por sensor | Sensor atascado (82.4%), ruido, inercia | Baja |
| Alta | M2 — Ratio PRAD/PRGINT y delta_temp | Correlación entre sensores invisible | Baja |
| Alta | M3 — Percentiles contextuales por Mes+Hora | Anomalías estacionales no detectadas | Media |
| Alta | M6 — TimeSeriesSplit | Data leakage en validación | Baja |
| Media | M4 — Threshold adaptativo por sensor | Corrección excesiva en sensores estables | Baja |
| Media | M5 — Corrección condicional a confianza | Modificar valores que no son anómalos | Media |
| Media | M7a — Isolation Forest complementario | Anomalías no vistas en entrenamiento | Media |
| Media | M7e — STL + residuos | Estacionalidad en detección de spikes | Media |
| Baja | M7b — LSTM Autoencoder | Captura temporal y correlaciones sin feature engineering | Alta |
| Baja | M7d — Matrix Profile | Sensor atascado y discords temporales | Media |
| Baja | M8 — Búsqueda de hiperparámetros | Modelo no optimizado | Media |
| Baja | M9 — Relajar CO2 contextual | 0 secuencias de ese tipo generadas | Baja |
