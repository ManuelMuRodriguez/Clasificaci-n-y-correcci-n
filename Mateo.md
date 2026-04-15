# Estado Original del Sistema — Referencia Mateo

Este documento recoge cómo estaba el sistema **antes de los cambios realizados para la publicación científica**. Sirve de referencia para entender qué se ha modificado, por qué, y cuál era el punto de partida.

El notebook de referencia original es:
`Ultimo antes del cambio Clasificación_y_Corrección_de_Anomalías.ipynb`

---

## Dataset original

| Propiedad | Valor |
|---|---|
| Fichero | `2020_10_10-2020_12_29.csv` |
| Registros | 233,381 filas |
| Columnas | 13 |
| Período | 10/10/2020 al 29/12/2020 |
| Duración | ~3 meses (sin estacionalidad completa) |
| Frecuencia | ~5 minutos |
| Formato de fecha | `%d/%m/%Y %H:%M:%S` |
| Origen | Datos históricos de invernadero, no AGROCONNECT |

El dataset cubría solo otoño-invierno, sin verano ni primavera. Esto limitaba la capacidad del sistema para generalizar a todas las estaciones.

---

## Variables y columnas

Las mismas 12 variables del paper (PCO2EXT, PHEXT, PRAD, PRGINT, PTEXT, PVV, UVENT_cen, UVENT_lN, XCO2I, XHINV, XTINV, XTS) más la columna `Fecha`.

Las ventanas de ventilación (`UVENT_cen`, `UVENT_lN`) venían ya agregadas en el CSV original, no se calculaban desde columnas individuales `_POS`.

---

## Rangos válidos para "Valores Fuera de Rango"

**Los rangos no eran límites físicos del hardware. Eran estadísticos, calculados automáticamente sobre el propio dataset:**

```python
# Por cada sensor, se calculan los percentiles 0.1 y 99.9 del dataset
limites_automaticos[col] = {
    'min': p0_1,    # percentil 0.1 del conjunto de datos
    'max': p99_9    # percentil 99.9 del conjunto de datos
}
rangos_validos_sensores = limites_automaticos
```

La inyección de anomalías "Fuera de Rango" introducía valores un **10% más allá** de esos límites estadísticos:

```python
margen_factor = 0.1  # 10% del span del rango

# Ejemplo para PTEXT si P0.1 = 3°C y P99.9 = 28°C (span = 25°C):
# valor anómalo alto  = 28 + (25 × 0.1) = 30.5°C
# valor anómalo bajo  =  3 - (25 × 0.1) = 0.5°C
```

**Problema:** un percentil calculado sobre 3 meses de otoño-invierno no es equivalente a los límites físicos del sensor. Un valor de 35°C en PTEXT en verano es perfectamente normal pero estaría fuera del P99.9 del periodo original, generando falsos positivos.

---

## Features temporales

Solo se extraían **3 features** del campo `Fecha`:

```python
Hora        # hora del día (0-23)
DiaSemana   # día de la semana (0-6)
Mes         # mes del año (1-12)
```

**No existían:**
- Ventanas deslizantes (`rolling_std`, `rolling_mean`)
- Diferencias con instantes anteriores (`diff`)
- Valores retrasados (`lag_1`, `lag_N`)
- Features cruzadas entre sensores

El modelo veía cada instante de forma **casi independiente**. La única "memoria" temporal eran esas 3 columnas de calendario.

---

## Modelo de detección y clasificación

**Arquitectura:** Random Forest (scikit-learn)

- **Detección binaria:** clasifica cada muestra como `normal` o `anomalía`
- **Clasificación multiclase:** identifica el tipo entre los 6 definidos

Los modelos se guardaban con `joblib` al final del notebook:

```python
import joblib
joblib.dump(modelo_deteccion,     'models/modelo_deteccion.pkl')
joblib.dump(modelo_clasificacion, 'models/modelo_clasificacion.pkl')
```

**Resultados publicados en el paper:**

| Métrica | Detección (binaria) | Clasificación (tipo) |
|---|---|---|
| Accuracy | ~97% | ~82% |
| F1-score | ~96% | ~80% |

---

## Correlaciones entre sensores

El notebook **calculaba la matriz de correlación** entre variables, pero esta información **no se usaba como feature del modelo**. Solo se usaba para guiar la inyección de anomalías de tipo "Desviación de Correlación" (introducir valores que rompan la correlación esperada entre dos sensores).

Relaciones físicas reales que el modelo ignoraba por no tener features cruzadas:
- `PTEXT` ↔ `XTINV` (temperatura exterior influye en la interior)
- `PRAD` ↔ `PRGINT` (radiación exterior proporcional a la interior)
- `UVENT_cen` / `UVENT_lN` → `XTINV`, `XCO2I`, `XHINV` (ventilación regula el microclima)
- `XTS` sigue a `XTINV` con horas de retraso (inercia térmica del suelo)

---

## Frecuencia de muestreo

El paper menciona una frecuencia de **~5 minutos**. El resampleado original usaba media (`resample("5min").mean()`).

**Problema identificado:** con media de 10 muestras de 30 s, un spike de un solo registro quedaba atenuado al ~10% de su magnitud original, dificultando la detección de ruido y valores fuera de rango transitorios.

---

## Tipos de anomalías inyectadas

| Tipo | Descripción | Inyección original |
|---|---|---|
| Datos faltantes | NaN en una o varias variables | Sustitución directa por NaN |
| Sensor atascado | Valor constante repetido N veces | Repetir el mismo valor 5–20 intervalos |
| Ruido | Oscilaciones rápidas aleatorias | Ruido gaussiano escalado |
| Fuera de rango | Valor fuera de P0.1/P99.9 ± 10% | Valor sintético fuera del percentil estadístico |
| Desviación de correlación | Rompe la correlación entre dos sensores | Valor inconsistente con el sensor correlacionado |
| Contextual | Valor plausible en otro contexto (ej. temperatura alta de noche) | Valor real de otra franja horaria |

---

## Corrección de anomalías

Las correcciones eran por regla según el tipo detectado:

| Tipo | Método de corrección |
|---|---|
| Datos faltantes | Imputación por KNN o media móvil |
| Sensor atascado | Interpolación lineal o promedio estacional |
| Ruido | Filtro de media móvil |
| Fuera de rango | Truncar al límite estadístico más cercano |
| Desviación de correlación | Promedio ponderado con el sensor correlacionado |
| Contextual | Promedio estacional (misma hora, mismo mes) |

---

## Resumen de limitaciones identificadas

| Limitación | Impacto |
|---|---|
| Dataset de solo 3 meses (otoño-invierno) | Sin estacionalidad completa, modelos no generalizan a verano |
| Rangos "Fuera de Rango" estadísticos, no físicos | Falsos positivos en verano; no refleja los límites reales del hardware |
| Sin features temporales (rolling, diff, lag) | No detecta sensor atascado por evolución temporal; spikes visibles solo si son muy grandes |
| Correlaciones no usadas como features | El modelo ignora relaciones físicas entre sensores |
| Resampleo a 5 min con media | Atenúa spikes al ~10% de su magnitud |
| Ventanas de ventilación ya agregadas en el CSV | No trazabilidad de las columnas individuales `_POS` |

---

## Cambios realizados respecto a este estado original

| Aspecto | Estado original (Mateo) | Estado actual |
|---|---|---|
| Dataset | `2020_10_10-2020_12_29.csv` (3 meses) | `2023_12_13-2024_12_31_1min.csv` (13 meses) |
| Frecuencia | ~5 minutos (media de 10 muestras) | 1 minuto (media de 2 muestras) |
| Filas | 233,381 | 552,982 |
| Rangos "Fuera de Rango" | P0.1/P99.9 del dataset | Límites físicos reales del hardware (fichas técnicas PLC) |
| Sensores exteriores | Columnas ya en el CSV | Mapeadas desde AGROCONNECT XLSX (415 ficheros) |
| Ventanas de ventilación | Ya agregadas en el CSV | Calculadas como media de 6 col. centrales + 7 col. laterales `_POS` |
| Primer dato válido ventanas | Desconocido | 2024-03-05 13:56:30 (fichero AGROCONNECT_20240305-193134.xlsx) |
| Dataset disponible sin NaN ventanas | — | 2024-03-06 → 2025-11-30 (20 meses limpios) |
| Rangos físicos documentados | No disponibles | Extraídos de `AGROCONNECT_Variables_PLCs.xlsx` |
| Semilla aleatoria inyección | No controlada (no reproducible) | `random_state=42` en todas las funciones |
| Rendimiento inyección | Bucles Python con `.loc` fila a fila — horas por celda | Vectorizado con numpy `iat`/`cumsum` — segundos por celda |
| Inyección Sensor Atascado | Bucle O(N×intentos), itera todos los índices | `cumsum` vectorizado O(N), valida toda la ventana sin NaN |
| Inyección CO2 sin respuesta | Bucle anidado O(N×intentos) sobre 552K filas | `cumsum` vectorizado con condiciones de contexto |
