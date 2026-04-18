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
| Dataset | `2020_10_10-2020_12_29.csv` (3 meses) | `scada_2024_03_06-2025_03_05_1min.csv` (12 meses) |
| Frecuencia | ~5 minutos (media de 10 muestras) | 1 minuto (media de 2 muestras) |
| Filas | 233,381 | 524,161 (SCADA) |
| Rangos "Fuera de Rango" | P0.1/P99.9 del dataset | Límites físicos reales del hardware (fichas técnicas PLC) |
| Sensores exteriores | Columnas ya en el CSV | Mapeadas desde AGROCONNECT XLSX (714 ficheros) en `Dataset/SCADA/` |
| Ventanas de ventilación | Ya agregadas en el CSV | Calculadas como media de 6 col. centrales + 7 col. laterales `_POS` |
| Primer dato válido ventanas | Desconocido | 2024-03-05 13:56:30 (fichero AGROCONNECT_20240305-193134.xlsx) |
| Dataset disponible sin NaN ventanas | — | 2024-03-06 → 2025-11-30 (20 meses limpios) |
| Rangos físicos documentados | No disponibles | Extraídos de `AGROCONNECT_Variables_PLCs.xlsx` |
| Semilla aleatoria inyección | No controlada (no reproducible) | `random_state=42` en todas las funciones |
| Rendimiento inyección | Bucles Python con `.loc` fila a fila — horas por celda | Vectorizado con numpy `iat`/`cumsum` — segundos por celda |
| Inyección Sensor Atascado | Bucle O(N×intentos), itera todos los índices | `cumsum` vectorizado O(N), valida toda la ventana sin NaN |
| Inyección CO2 sin respuesta | Bucle anidado O(N×intentos) sobre 552K filas | `cumsum` vectorizado con condiciones de contexto |
| Timestamps SCADA | — | Convertidos a UTC (se resta el offset +1H/+2H DST del sufijo) |
| Segunda fuente de datos | No existía | OPC UA (TXT, 503 ficheros en `Dataset/OPCUA/`) desde julio 2024 |
| Dataset combinado | No existía | `combined_2024_03_06-2025_03_05_1min.csv` — SCADA + OPC UA, NaN rellenados |
| NaN SCADA vs OPC UA | — | SCADA 14.3% / OPC UA 3.1% (período solapado jul 2024–mar 2025) |
| Mapeo XTS en OPC UA | — | `OPC_INVER_TEMP_SUELO5_S1` — S1 OPC UA = S1 SCADA (MAE=0.017°C, verificado con comparar_xts_opcua.py) |

---

## Fusión de fuentes SCADA + OPC UA (dataset Combined)

### Motivación

SCADA tiene más cobertura temporal (desde mar 2024) pero 14.3% de NaN. OPC UA tiene menos NaN (3.1%) pero solo desde julio 2024. La fusión combina lo mejor de ambas fuentes.

### Proceso de generación

```bash
python src/prepare_dataset.py --source combined
# Genera: data/combined_2024_03_06-2025_03_05_1min.csv
```

### Algoritmo de fusión (`preparar_dataset_combined` en `src/prepare_dataset.py`)

1. **Cargar SCADA** → renombrar a nombres del paper → resamplear a 1 min
2. **Cargar OPC UA** → renombrar a nombres del paper → resamplear a 1 min
3. **SCADA como base** — índice de tiempo del SCADA (mar 2024 → mar 2025)
4. **Alinear OPC UA** al índice de SCADA con `reindex`
5. **Rellenar NaN** variable a variable: donde SCADA tiene NaN y OPC UA tiene dato, se usa OPC UA — excepto en los períodos excluidos (ver abajo)
6. Los NaN que persisten son huecos en ambas fuentes simultáneamente

### Correcciones aplicadas antes de la fusión

| Corrección | Detalle |
|---|---|
| Timestamps SCADA → UTC | Se resta el offset `+1H`/`+2H DST` del sufijo de cada timestamp |
| Mapeo XTS en OPC UA | `OPC_INVER_TEMP_SUELO5_S1` — S1 OPC UA = S1 SCADA (MAE=0.017°C, verificado con `comparar_xts_opcua.py`) |
| Período malo XTS en OPC UA | 2024-07-18 → 2024-08-04: sensor físicamente mal conectado — OPC UA excluido para XTS en ese rango |

### Períodos excluidos de OPC UA (`OPCUA_PERIODOS_EXCLUIDOS` en `prepare_dataset.py`)

Algunos sensores OPC UA tienen datos incorrectos en determinados períodos (mala conexión física, offset de calibración, etc.). Estos períodos se excluyen del relleno de NaN — en esas fechas, si SCADA tiene NaN, el dato queda como NaN en el dataset combinado en lugar de rellenarse con un valor incorrecto de OPC UA.

Los ficheros raw de OPC UA **no se modifican** — la exclusión se aplica solo en el pipeline de generación del dataset combinado.

| Variable | Período excluido | Motivo | Verificación |
|---|---|---|---|
| XTS | 2024-07-18 → 2024-08-04 | Sensor `OPC_INVER_TEMP_SUELO5_S1` mal conectado físicamente | MAE diario >1°C hasta el 4 ago; cae a 0.03°C el 5 ago |

#### Herramienta: `src/comparar_xts_opcua.py`

Compara todas las columnas `TEMP_SUELO` (excluyendo `SUELO30`) de los ficheros raw de SCADA y OPC UA. Carga los ficheros originales — no los CSV procesados — para tener acceso a todas las columnas sin filtrar.

**Opciones de ejecución:**

```bash
# Un mes concreto — abre gráfica interactiva y guarda en data/comparacion_xts_opcua.png
python src/comparar_xts_opcua.py --mes 2024-10

# Todos los meses con OPC UA disponible — guarda una PNG por mes en data/suelo_meses/, sin ventanas
python src/comparar_xts_opcua.py --loop

# Cambiar carpeta de salida del loop
python src/comparar_xts_opcua.py --loop --output-dir data/mi_carpeta

# Usar rutas distintas a las por defecto
python src/comparar_xts_opcua.py --mes 2024-10 --scada-dir Dataset/SCADA --opcua-dir Dataset/OPCUA
```

**Argumento** | **Por defecto** | **Descripción**
---|---|---
`--mes YYYY-MM` | — | Mes concreto a comparar (mutuamente exclusivo con `--loop`)
`--loop` | — | Itera todos los meses con datos OPC UA disponibles
`--output-dir` | `data/suelo_meses/` | Carpeta de salida en modo `--loop`
`--scada-dir` | `Dataset/SCADA` | Carpeta con los ficheros `AGROCONNECT_*.xlsx`
`--opcua-dir` | `Dataset/OPCUA` | Carpeta raíz con los ficheros `OPC_*.txt`

**Salida por mes:**
- Tabla en consola con MAE (°C) y correlación de cada columna OPC UA vs SCADA S1
- Gráfica con todas las series superpuestas: SCADA en negro/gris, OPC UA en colores
- La leyenda inferior incluye el MAE y correlación de cada candidato

#### Cómo se verificó el período malo de XTS

El script se usó para detectar qué columna OPC UA corresponde a `INVER_TEMP_SUELO5_S1` de SCADA y en qué período el sensor estaba mal conectado. Con esto se detectó:

1. **OPC UA S1 = SCADA S1** — el mapeo S1→S1 es correcto (MAE=0.017°C en meses correctos)
2. **Período malo**: julio 2024 y primeros días de agosto presentaban MAE de 4-5°C — el sensor estaba físicamente mal conectado a otro canal

Para confirmar la fecha exacta de corte se calculó el MAE día a día:

```
2024-08-03    4.02°C   ← malo
2024-08-04    1.19°C   ← malo
2024-08-05    0.03°C   ← correcto (sensor reconectado)
```

Corte definitivo: **2024-08-04 23:59:59** → OPC UA válido desde el 5 de agosto 00:00 UTC.

---

## Variables de ventilación (UVENT_cen, UVENT_lN) — Estado y fiabilidad

### Contexto técnico (fuente: Paco, 15/04/2026)

Las variables de ventilación tienen una historia compleja que afecta directamente a su fiabilidad:

**Hasta enero 2026 — lazo abierto sin encoders:**
- No existían encoders físicos que confirmaran la posición real de las ventanas
- El sistema funcionaba en lazo abierto: se enviaba una consigna de posición pero no había realimentación de la posición real

**Qué guarda cada fuente:**

| Fuente | Variable | Qué representa | Fiabilidad |
|---|---|---|---|
| SCADA | `UVCEN*_POS`, `UVLAT*_POS` | Posición *comandada* al actuador — lo que se pidió abrir | No es posición real |
| OPC UA | `OPC_..._POS` | Posición de escritura del control SCADA — igual que SCADA | No es posición real |
| OPC UA | `OPC_..._POS_VALOR` | Posición de lectura — en algunos períodos encoder real, en otros ensayos MATLAB | Depende del período |

**Encoders en OPC UA:** los encoders se fueron instalando progresivamente a lo largo de 2024-2025, **dentro del rango del dataset actual**. Sus columnas siguen el patrón `OPC_INVER_<ventana>_Encoder` y solo están disponibles en OPC UA (no en SCADA).

| Encoder | Primera aparición | Primer valor real |
|---|---|---|
| `OPC_INVER_UVLAT1N_Encoder` | 2024-11-16 | **2024-11-16** |
| `OPC_INVER_UVCEN11_Encoder` | 2025-05-07 | 2025-05-07 |
| `OPC_INVER_UVCEN12_Encoder` | 2025-05-07 | 2025-05-07 |
| `OPC_INVER_UVCEN13_Encoder` | 2025-05-07 | 2025-05-07 |
| `OPC_INVER_UVCEN21_Encoder` | 2025-05-07 | 2025-05-07 |
| `OPC_INVER_UVCEN22_Encoder` | 2025-05-07 | 2025-05-08 |
| `OPC_INVER_UVCEN23_Encoder` | 2025-05-07 | 2025-05-08 |
| `OPC_INVER_UVLAT1NO_Encoder` | 2025-05-07 | 2025-05-07 |
| `OPC_INVER_UVLAT1S_Encoder` | 2025-05-07 | 2025-05-07 |
| `OPC_INVER_UVLAT1SO_Encoder` | 2025-05-07 | **2025-06-02** (tardó en dar valores reales) |
| `OPC_INVER_UVLAT2E_Encoder` | 2025-05-07 | 2025-05-07 |
| `OPC_INVER_UVLAT2N_Encoder` | 2025-05-07 | 2025-05-07 |
| `OPC_INVER_UVLAT2S_Encoder` | 2025-05-07 | 2025-05-08 |

Esto significa que **desde mayo 2025 tenemos posición real de todas las ventanas** en el dataset, y desde nov 2024 para UVLAT1N.

### Implicaciones para el modelo de detección de anomalías

**Las ventanas NO se pueden usar como variable objetivo de detección** — no hay forma de saber si el valor registrado es anómalo sin conocer la posición real.

**Sí tienen valor como variable contextual:** aunque la posición sea comandada (no real), permite detectar anomalías contextuales de otros sensores. Ejemplos:
- UVENT = 0% (ventanas cerradas) + XTINV = 50°C en verano → correlación anómala
- UVENT = 100% (ventanas abiertas) + XCO2I = 2000 ppm → correlación anómala

### Estado del mapeo actual en el pipeline

Se usa `_POS_VALOR` (no `_POS`) en OPC UA, verificado con correlación vs SCADA:

| Campo OPC UA | MAE vs SCADA | Correlación | Conclusión |
|---|---|---|---|
| `_POS` | 47.69% | 0.00 | Valor binario (0/100), no útil |
| `_POS_VALOR` | 4.04% | 0.92 | Correlaciona bien con la consigna SCADA |

El MAE residual de ~4% entre `_POS_VALOR` y SCADA refleja que son dos formas distintas de registrar la misma consigna, no la posición real.

### Ensayos registrados — `Dataset/Ensayos/Lista de ensayos Invernadero AgroConnect.xlsx`

Lista completa de días con experimentos controlados en el invernadero. Dentro del rango del dataset (2024-03-06 → 2025-11-30):

| Período | Tipo de ensayo | Implicación |
|---|---|---|
| 2024-03-18 → 2024-03-25 | Ventilación | UVENT activamente controlada — datos más fiables |
| 2024-04-01 → 2024-04-05 | Ventilación | UVENT activamente controlada |
| 2024-04-23 → 2024-04-24 | Ventilación | UVENT activamente controlada |
| 2024-11-18 → 2024-11-24 | Ventilación | UVENT activamente controlada |
| 2025-01-13 → 2025-02-26 | Calefacción biomasa | XTINV/XTS pueden mostrar valores inusualmente altos |
| 2025-03-13 → 2025-03-27 | Deshumidificación | XHINV puede mostrar valores inusualmente bajos |
| 2025-04-04 | Calefacción biomasa | XTINV/XTS afectados |
| 2025-04-19 | Ventilación | UVENT activamente controlada |
| 2025-06-12 → 2025-06-24 | Ventilación | UVENT activamente controlada |
| 2025-07-22 | CO2 | XCO2I/PCO2EXT pueden mostrar valores inusualmente altos |
| 2025-09-17, 24, 30 | Ventilación | UVENT activamente controlada |
| 2025-11-19 → 2025-11-20 | Ventilación | UVENT activamente controlada |

### Implicaciones para el modelo

**Problema:** durante los días de ensayo, los sensores muestran comportamiento intencionalmente forzado que el modelo puede clasificar como anomalía cuando en realidad es un experimento controlado.

**Solución:** añadir una columna `ensayo` al dataset combinado con el tipo de ensayo activo (`ventilacion`, `calefaccion`, `deshumidificacion`, `co2`, o vacío). Esto permite:
- Excluir días de ensayo del entrenamiento del modelo de detección
- O etiquetarlos como clase separada (`anomalia_ensayo`) para que el modelo los distinga
- Usar los ensayos de ventilación para validar correlaciones UVENT↔temperatura/humedad/CO2

El fichero de ensayos se carga en `src/prepare_dataset.py` al generar el combined.

**Insight clave sobre ventilación (fuente: Paco, 15/04/2026):**

Durante los ensayos de ventilación, el sistema estaba funcionando correctamente y supervisado activamente por el equipo. Esto significa que los datos de UVENT registrados por **ambas fuentes — SCADA y OPC UA — son fiables y consistentes** en esos períodos. Son los únicos días del dataset (anterior a enero 2026) donde se puede confiar plenamente en los valores de ventilación.

| Tipo de día | Fiabilidad UVENT (SCADA) | Fiabilidad UVENT (OPC UA) |
|---|---|---|
| **Días de ensayo de ventilación** | Alta — sistema supervisado y funcionando correctamente | Alta — ídem |
| Operación normal (antes de ene 2026) | Media — posición comandada sin encoders reales | Media — ídem |
| Ensayos de calefacción/CO2/deshumidificación | Media — ventanas como variable secundaria | Media — ídem |

Esto hace que los días de ensayo de ventilación sean los mejores ejemplos de **comportamiento correcto conocido** de UVENT para entrenar o validar el modelo, tanto en valores absolutos como en correlaciones UVENT↔XTINV/XHINV/XCO2I.

### Conclusión del análisis cuantitativo (_POS vs _POS_VALOR vs Encoder)

**Script:** `src/comparar_tipos_ventilacion.py` — genera gráficas por grupo y mes, y un CSV de métricas en `data/tipos_ventilacion/resumen_metricas.csv`.

El análisis de MAE respecto al encoder físico (dic 2024 → dic 2025) confirma que ninguna de las señales software refleja fielmente la posición real de la ventana:

| Señal | MAE típico vs Encoder | Interpretación |
|---|---|---|
| `_POS_VALOR` (realimentación software) | **40–85 %** en la mayoría de ventanas | No fiable salvo UVLAT1N (< 2 %) |
| `_POS` (comando al actuador) | **80–99 %** de forma sistemática | El actuador no ejecuta fielmente la orden |

**UVLAT1N** es la única ventana con comportamiento sostenidamente fiable, probablemente por ser el encoder más antiguo y mejor calibrado.

**Implicación para el modelo de detección de anomalías:**

Las señales de posición de ventana (`_POS`, `_POS_VALOR`) se excluyen del modelo durante el período de estudio (2023-2024), ya que los encoders físicos no estaban disponibles y las señales software demuestran desviaciones no asumibles. El efecto de la ventilación queda recogido implícitamente en las variables climáticas interiores (`XTINV`, `XHINV`, `XCO2I`), que actúan como proxies del estado real de apertura. Esta limitación se documenta en el paper y abre la puerta a incorporar la posición real del encoder en trabajo futuro.

### Cobertura temporal por fuente

| Fuente | Período | Filas (1 min) | NaN medio |
|---|---|---|---|
| SCADA | 2024-03-06 → 2025-11-29 | 912,900 | 14.3% |
| OPC UA | 2024-07-18 → 2025-11-29 | 719,122 | 7.8% |
| Combined | 2024-03-06 → 2025-11-29 | 912,900 | 4.5% |

### Variables del paper y sus columnas fuente

| Variable | SCADA | OPC UA |
|---|---|---|
| PCO2EXT | `CO2_EXTERIOR_10M` | `OPC_CO2_EXTERIOR_10M` |
| PHEXT | `HR_EXTERIOR_10M` | `OPC_HR_EXTERIOR_10M` |
| PRAD | `RADGLOBAL_EXTERIOR_10M` | `OPC_RADGLOBAL_EXTERIOR_10M` |
| PRGINT | `INVER_RADGLOBAL_INTERIOR_S1` | `OPC_INVER_RADGLOBAL_INTERIOR_S1` |
| PTEXT | `TEMP_EXTERIOR_10M` | `OPC_TEMP_EXTERIOR_10M` |
| PVV | `VELVIENTO_EXTERIOR_10M` | `OPC_VELVIENTO_EXTERIOR_10M` |
| UVENT_cen | media de 6 columnas `_POS` centrales | media de 6 `OPC_..._POS` centrales |
| UVENT_lN | media de 7 columnas `_POS` laterales | media de 7 `OPC_..._POS` laterales |
| XCO2I | `INVER_CO2_INTERIOR_S1` | `OPC_INVER_CO2_INTERIOR_S1` |
| XHINV | `INVER_HR_INTERIOR_S1` | `OPC_INVER_HR_INTERIOR_S1` |
| XTINV | `INVER_TEMP_INTERIOR_S1` | `OPC_INVER_TEMP_INTERIOR_S1` |
| XTS | `INVER_TEMP_SUELO5_S1` | `OPC_INVER_TEMP_SUELO5_S1` (verificado: MAE=0.017°C vs S1 SCADA) |

---

## Versiones del notebook para comparación experimental

### Fichero de datos común a v2 y v3

| Parámetro | Valor |
|---|---|
| Fichero | `data/combined_2024_03_06-2025_11_30_1min.csv` |
| Período usado | 2024-03-06 → 2025-03-07 (1 año completo — estacionalidad completa) |
| Filas tras filtro | ~525,600 (1 min × 12 meses) |
| NaN medio | ~4.5% (SCADA + OPC UA fusionados) |
| Frecuencia | 1 minuto |

Se usa el dataset `combined` (y no el SCADA o OPC UA por separado) porque es el de menor NaN al fusionar ambas fuentes. El período se limita a un año exacto para garantizar estacionalidad completa y comparación justa entre versiones.

---

### v2 — Baseline actualizado (`Clasificación_y_Corrección_v2.ipynb`)

Notebook de referencia con los datos nuevos pero sin mejoras metodológicas. Permite comparar directamente con v3 aislando el efecto de las mejoras.

**Diferencias respecto al notebook original de Mateo:**

| Aspecto | Original Mateo | v2 |
|---|---|---|
| Dataset | `2020_10_10-2020_12_29.csv` (3 meses, 5 min) | `combined` (12 meses, 1 min) |
| Fuente datos | CSV histórico | SCADA + OPC UA fusionados |
| Rangos "Fuera de Rango" | P0.1/P99.9 estadísticos | P0.1/P99.9 estadísticos (igual) |
| Features | 15 columnas puntuales | 15 columnas puntuales (igual) |
| Split | Aleatorio 70/30 | Aleatorio 70/30 (igual) |
| UVENT en modelo | Sí | Sí (igual) |
| Inyección vectorizada | No | Sí |
| random_state | No controlado | 42 en todo |

---

### v3 — Versión mejorada para publicación (`Clasificación_y_Corrección_v3.ipynb`)

Notebook con todas las mejoras metodológicas documentadas. El objetivo es demostrar que las mejoras incrementan las métricas de forma significativa respecto a v2.

#### Mejoras implementadas

**UVENT — Exclusión de señales de ventilación no verificables**

`UVENT_cen` y `UVENT_lN` se excluyen tanto de la inyección de anomalías como de las features del modelo. La señal `_POS/_POS_VALOR` tiene un MAE del 40–85% respecto al encoder físico (análisis en `src/comparar_tipos_ventilacion.py`, CSV en `data/tipos_ventilacion/resumen_metricas.csv`). El efecto real de la ventilación queda recogido implícitamente a través de las features cruzadas M2 (`gradiente_co2`, `delta_temp`, `delta_hum`).

> *"Las variables de posición de ventana fueron excluidas del modelo por carecer de realimentación física verificable mediante encoder en el período de estudio. Su efecto queda recogido implícitamente a través de los gradientes térmico, hídrico y de CO2 interior-exterior."*

---

**M1 — Features temporales (rolling, diff, lag, z-score local)**

El modelo v2 ve cada minuto de forma aislada — sus únicas referencias temporales son Hora, DiaSemana y Mes. Esto lo hace ciego a patrones que solo se detectan en la evolución temporal, especialmente el Sensor Atascado (82.4% recall en v2).

Features añadidas por cada sensor:

| Feature | Ventana | Qué detecta |
|---|---|---|
| `{col}_rmean_30m` | 30 min | Tendencia reciente |
| `{col}_rstd_30m` | 30 min | `≈ 0` sostenido → sensor atascado |
| `{col}_rmean_3h` | 3 h | Tendencia larga |
| `{col}_rstd_3h` | 3 h | Comparación con variabilidad de corto plazo |
| `{col}_diff1` | 1 min | Cambio brusco → spike |
| `{col}_diff30` | 30 min | Cambio lento → tendencia anómala |
| `{col}_lag1` | 1 min | Valor anterior inmediato |
| `{col}_lag30` | 30 min | Valor hace media hora |
| `{col}_zscore_local` | 30 min | Desviaciones respecto a la tendencia propia |

Total features nuevas M1: ~90 columnas (9 derivadas × 10 sensores).

---

**M2 — Features cruzadas entre sensores**

El modelo v2 no puede detectar anomalías que solo se manifiestan cuando se rompe la relación física entre dos sensores. M2 hace explícitas las relaciones más importantes del invernadero:

| Feature | Fórmula | Relación física |
|---|---|---|
| `ratio_rad` | `PRGINT / (PRAD + 1)` | Factor de transmisión del film de polietileno. Normal: 0.3–0.7. Varía con blanqueo |
| `delta_temp` | `XTINV − PTEXT` | Diferencia temperatura interior−exterior. Alta en verano sin ventilación |
| `delta_hum` | `XHINV − PHEXT` | Diferencia humedad interior−exterior. Tiende a 0 con ventilación abierta |
| `gradiente_co2` | `XCO2I − PCO2EXT` | CO2 interior−exterior. Captura efecto real de ventilación |
| `delta_temp_suelo` | `XTS − XTINV` | Inercia térmica del suelo. Cambio brusco en 1 min = físicamente imposible |

**Nota sobre `ratio_rad` y blanqueo (invernadero de Almería):**
El film de polietileno se blanquea con carbonato cálcico (CaCO₃) en los meses de mayor radiación (típicamente mayo–septiembre) para reducir el calentamiento interior. Esto reduce el `ratio_rad` de forma estacional (~0.5–0.7 sin blanquear, ~0.3–0.5 blanqueado). Las features contextuales M3 (percentiles por Mes×Hora) capturan este patrón automáticamente. Ante la ausencia de registros de aplicación de blanqueo, se plantea como trabajo futuro la detección automática mediante la mediana móvil de 7 días del `ratio_rad`.

---

**M3 — Rangos físicos reales del hardware**

En v2 los rangos para la inyección de "Valores Fuera de Rango" se calculan como P0.1/P99.9 del propio dataset — rangos estadísticos que dependen de las condiciones del año concreto. Un valor de 39°C en PTEXT en verano podría etiquetarse como anomalía aunque el sensor funcione perfectamente (su rango físico llega a 60°C).

En v3 se usan los rangos físicos certificados del fabricante:

| Variable | Rango físico real | Sensor |
|---|---|---|
| PCO2EXT, XCO2I | 0 – 2000 ppm | E+E Elektronik EE820 |
| PHEXT, XHINV | 0 – 100 % | Campbell HC2A-S3 |
| PRAD, PRGINT | 0 – 2000 W/m² | Campbell SP-214/SP-110 |
| PTEXT, XTINV | -40 – 60 °C | Campbell HC2A-S3 |
| PVV | 0 – 45.8 m/s | Wittich & Visser PA2 |
| XTS | -50 – 70 °C | Campbell 109 NTC |

Solo se etiqueta como "Fuera de Rango" lo que es físicamente imposible para ese hardware.

---

**M4 — Umbrales de corrección adaptativos por sensor**

En v2 todos los métodos de corrección usan `factor = 2.0 × std` para todos los sensores, independientemente de su variabilidad natural. En v3 el factor se calcula por el coeficiente de variación (CV = std / media):

| CV | Factor | Sensores típicos |
|---|---|---|
| < 0.10 (muy estable) | 1.5 | XTS, PCO2EXT |
| 0.10 – 0.50 (medio) | 2.0 | XTINV, XHINV, PHEXT |
| > 0.50 (muy variable) | 2.5 | PRAD, PVV, PRGINT |

---

**M6 — TimeSeriesSplit k=4 en lugar de split aleatorio**

En v2 el split es aleatorio estratificado (70/30). En series temporales esto provoca data leakage: el modelo puede entrenarse con datos de noviembre y testearse con agosto, inflando artificialmente las métricas. En v3 se usa TimeSeriesSplit con 4 folds que avanzan en el tiempo:

```
Fold 1 — Train: mar→jul 2024    Test: ago→sep 2024   (verano)
Fold 2 — Train: mar→sep 2024    Test: oct→nov 2024   (otoño)
Fold 3 — Train: mar→nov 2024    Test: dic 2024        (invierno temprano)
Fold 4 — Train: mar→dic 2024    Test: ene→mar 2025   (invierno pleno)
```

Las métricas finales son la media ± std de los 4 folds, evaluando el modelo en todas las estaciones sin data leakage. El modelo del Fold 4 (el de mayor datos de train) se usa para la fase de corrección.

---

**M9 — Condiciones relajadas en inyección CO2 contextual**

En v2 la condición para inyectar "CO2 sin respuesta a ventilación" era demasiado restrictiva (ventilación > 70%, gradiente CO2 > 50 ppm, duración 2–4 min), generando 0 secuencias en el dataset. En v3 se relajan los umbrales:

| Parámetro | v2 | v3 |
|---|---|---|
| Umbral ventilación | > 70% | > 40% |
| Gradiente CO2 | > 50 ppm | > 30 ppm |
| Duración | 2–4 min | 3–8 min |

---

#### Resumen comparativo v2 vs v3

| Aspecto | v2 | v3 |
|---|---|---|
| Dataset | combined, 1 año | combined, 1 año (igual) |
| Features | 15 columnas puntuales | ~110 (+ rolling, diff, lag, cruzadas) |
| UVENT en modelo | Sí | No (excluida) |
| Rangos "Fuera de Rango" | P0.1/P99.9 estadísticos | Rangos físicos reales PLC |
| Split validación | Aleatorio 70/30 | TimeSeriesSplit k=4 (temporal) |
| Umbrales corrección | 2×std fijo | 1.5/2.0/2.5×std por CV del sensor |
| CO2 contextual | 0 secuencias generadas | Condiciones relajadas |
| Normalización | No (RF no la necesita) | No (ídem) |

---

#### Resultados reales v2 (dataset combinado 2024-03-06 → 2025-03-07)

##### Modelo 1 — Detector Normal vs Anomalía

| Métrica | Valor |
|---|---|
| Accuracy | 98.85% |
| Recall anomalía | 94% |
| Precision anomalía | 91% |
| F1 anomalía | 93% |
| ROC AUC | 0.9946 |
| PR AUC | 0.9813 |

Detección por tipo de anomalía:

| Tipo | Total test | Detectadas | No detectadas | Tasa detección |
|---|---|---|---|---|
| Datos Faltantes | 3214 | 3104 | 110 | 96.6% |
| Valores Fuera de Rango | 2944 | 2831 | 113 | 96.2% |
| Desviación Correlación | 749 | 710 | 39 | 94.8% |
| Ruido | 4663 | 4344 | 319 | 93.2% |
| Contextual | 326 | 293 | 33 | 89.9% |
| **Sensor Atascado** | **206** | **116** | **90** | **56.3% ← problema** |

Top features por importancia: XTINV (9%), PRAD (7.9%), PRGINT (7.6%), XCO2I (6.1%), XHINV (6%).
Los missing indicators contribuyen ~2% cada uno — informativos pero no decisivos.
UVENT_cen + UVENT_lN suman 8.2% — en v3 se eliminan, ese espacio lo ocupan features rolling y cross-sensor.

##### Modelo 2 — Clasificador de tipos de anomalía

| Tipo | Precision | Recall | F1 |
|---|---|---|---|
| Contextual | 1.00 | 0.94 | 0.97 |
| Datos Faltantes | 0.96 | 0.97 | 0.97 |
| Desviación Correlación | 0.97 | 0.99 | 0.98 |
| Ruido | 0.86 | 0.91 | 0.89 |
| Sensor Atascado | 0.97 | 0.97 | 0.97 |
| **Valores Fuera de Rango** | **0.86** | **0.77** | **0.81 ← problema** |
| **Global** | | | **90% accuracy** |

Confusión principal: 623 "Fuera de Rango" clasificadas como "Ruido" (21%) y 322 "Ruido" como "Fuera de Rango".
Causa: sin features temporales, el modelo no distingue un pico puntual de un valor físicamente fuera de rango.

> **Nota Sensor Atascado en Modelo 2:** el recall 97% es engañoso — Modelo 2 solo ve lo que Modelo 1 le pasa.
> Como Modelo 1 solo detectó el 56.3%, el cuello de botella real sigue siendo la detección.

#### Métricas clave a comparar con v3

| Métrica | v2 real | v3 objetivo | Por qué cambia |
|---|---|---|---|
| Detección Accuracy | 98.85% | ≥ 98.85% | Base ya alta |
| ROC AUC | 0.9946 | ≥ 0.9946 | |
| **Sensor Atascado recall (M1)** | **56.3%** | **↑ significativo** | M1: rolling_std ≈ 0 en valores constantes |
| **Fuera de Rango F1 (M2)** | **0.81** | **↑** | M1+M2: contexto temporal distingue pico vs rango |
| Clasificación Accuracy | 90% | ↑ | M1+M2 dan contexto temporal y relacional |
| Varianza entre folds | — | σ pequeña = robusto | M6 lo revela |

Los dos KPIs principales a mejorar con v3: **Sensor Atascado recall (56.3%)** y **Fuera de Rango F1 (0.81)**.

---

## Pipeline del notebook v2 — Pasos detallados

### FASE 0 — Setup
1. Crear entorno virtual e instalar dependencias

### FASE 1 — Preparación de datos
2. Importar librerías
3. Cargar dataset (`combined_2024_03_06-2025_11_30_1min.csv`, filtrado hasta 2025-03-07)
4. Verificar datos faltantes (NaN por columna)
5. Pasos preliminares (limpiar columnas, definir sensores, etc.)

### FASE 2 — Inyección de anomalías sintéticas (ground truth)
6. **Datos Faltantes** — poner NaN artificiales
7. **Sensor Atascado** — repetir el mismo valor N minutos
8. **Ruido / Picos Aleatorios** — añadir ruido gaussiano
9. **Valores Fuera de Rango** — valores por encima/debajo de límites estadísticos (P0.1/P99.9)
10. **Desviación de Correlación** — sensor que se desvía respecto a sus pares correlacionados
11. **Contextual** — anomalía lógica: luz nocturna, CO2 sin respuesta a ventilación

### FASE 3 — Modelo 1: Detector Normal vs Anomalía
12. Preparación de features X e y binario (0=normal, 1=anomalía)
13. División temporal train/test (70/30)
14. Imputar NaN con `SimpleImputer(median, add_indicator=True)`
15. Entrenar **Random Forest clasificador binario**
16. Predicciones + métricas (precision, recall, F1, matriz confusión)
17. Importancia de features
18. Desglose de detección por tipo de anomalía

### FASE 4 — Modelo 2: Clasificador de tipo de anomalía
19. Preparar datos: solo filas detectadas como anomalía con su etiqueta de tipo
20. División train/test
21. Entrenar **Random Forest multiclase** (6 tipos)
22. Predicciones + métricas por tipo

### FASE 5 — Corrección
23. **Datos Faltantes** → IterativeImputer (imputación multivariada)
24. **Sensor Atascado** → Método híbrido: corrección estacional + interpolación lineal
25. **Ruido** → Reglas locales (vecinos anterior/siguiente)
26. **Fuera de Rango** → Reglas locales
27. **Desviación Correlación** → Reglas locales
28. **Contextual** → Reglas locales

### FASE 6 — Evaluación y conclusiones
29. Métricas de corrección (MAE antes vs después)
30. Observaciones y conclusiones

### FASE 7 — Inferencia sobre datos reales (pipeline de producción)

**Objetivo:** Aplicar los modelos entrenados sobre el dataset completo sin inyección de anomalías, para detectar y corregir anomalías reales del invernadero. No hay ground truth — lo que detecte el modelo son anomalías reales.

31. **Cargar dataset real** — `combined_2024_03_06-2025_11_30_1min.csv` desde `data/combined_*.csv`. Parseo de fechas y extracción de features temporales. Sin inyección de anomalías.
32. **Cargar modelos desde disco** — `imputador_nans.joblib`, `modelo_1_detector.joblib`, `features_modelo_1.joblib`, `modelo_2_clasificador.joblib`, `label_encoder_modelo_2.joblib`
33. **Modelo 1 — Detección** — Aplica imputador + RF detector. Etiqueta cada punto como `normal` o `anomalia`. Gráficas por sensor con puntos anómalos marcados.
34. **Visualización post-detección** — Gráficas de todas las variables con anomalías señaladas para inspección visual.
35. **Modelo 2 — Clasificación** — Para cada punto anómalo, asigna tipo: Datos Faltantes, Sensor Atascado, Ruido, Fuera de Rango, Desviación Correlación o Contextual. Gráficas coloreadas por tipo.
36. **Modelo 3 — Corrección (6 métodos):**
    - Datos Faltantes → IterativeImputer multivariado
    - Sensor Atascado → corrección estacional + interpolación lineal
    - Ruido → sustitución por vecinos (anterior/siguiente)
    - Fuera de Rango → sustitución por vecinos
    - Desviación Correlación → sustitución por vecinos
    - Contextual → sustitución por vecinos
37. **Evaluación numérica** — MAE antes y después de corrección por sensor y tipo.
38. **Visualización de métricas** — Gráficas comparativas MAE antes vs después.
39. **Desglose final** — Tabla y gráfica de tarta con conteo y porcentaje de cada tipo detectado. Guardado en `data/fase7/`.

> **Nota de validación:** Sin ground truth real, la validación se hace por inspección de experto
> (confirmar si las anomalías detectadas tienen sentido físico). La validación temporal completa
> se hará con datos de **resto de 2025 y 2026** — datos que el modelo no vio en entrenamiento,
> incluyendo estaciones y condiciones distintas.

> **Nota RMSE/MAE en Fase 7:** Las métricas de corrección calculadas en esta fase **no son interpretables como calidad**. En Fases 1-6 el MAE compara el valor corregido contra el valor limpio original (conocido porque lo inyectamos). En Fase 7 no existe ese ground truth — el MAE compara contra el propio valor anómalo real, midiendo solo cuánto cambió la corrección. La evaluación cuantitativa válida es exclusivamente la de Fases 1-6.

> **Nota columna `ensayo`:** El dataset combined incluye una columna de texto `ensayo` (`ventilacion`, `calefaccion`, etc.) que registra los días de experimentos controlados. Esta columna debe filtrarse con `.select_dtypes(include='number')` antes de pasar los datos al imputador y a los modelos RF, ya que scikit-learn no acepta columnas de texto. Ya está aplicado en ambos notebooks v2 y v3.

---

## Plan de validación cualitativa — Fase 7

Para validar que el modelo detecta anomalías reales con sentido físico, usar este flujo tras ejecutar Fase 7:

```python
# 1. Ver qué tipos y cuántas anomalías detectó
df = datasets_listos['combined_2024_03_06-2025_11_30_1min']
print(df['etiqueta_tipo_anomalia'].value_counts())

# 2. Buscar fechas concretas de un tipo específico para inspeccionar
df_stuck = df[df['etiqueta_tipo_anomalia'] == 'Sensor Atascado']
print(df_stuck['Fecha'].head(20))

# 3. Usar la celda de visualización con esa fecha para confirmar visualmente
```

El experto del dominio (Manuel / compañero del invernadero) revisa una muestra y confirma:
- ✓ "Ese día sí falló el sensor X" → verdadero positivo
- ✗ "Eso es comportamiento normal" → falso positivo

Esta validación cualitativa es la que da validez científica al sistema en el paper cuando no hay ground truth etiquetado.

### Resultados Fase 7 v2 — Anomalías reales detectadas (dataset combinado 2024-03-06 → 2025-03-07)

| Tipo | Cantidad | % anomalías | % total |
|---|---|---|---|
| Datos Faltantes | 2002 | 41.6% | 0.38% |
| Ruido | 1407 | 29.2% | 0.27% |
| Valores Fuera de Rango | 894 | 18.6% | 0.17% |
| Sensor Atascado | 433 | 9.0% | 0.08% |
| Contextual | 43 | 0.9% | 0.01% |
| **Total** | **4,811** | | **0.9%** |

**Observaciones:**
- **0.9% anomalías reales** sobre 527,041 puntos — porcentaje creíble para un invernadero real
- **Datos Faltantes domina (41.6%)** — coherente con el ~4.5% NaN del combined
- **Fuera de Rango: 894 timestamps** — los 894 puntos aparecen en TODOS los sensores simultáneamente porque el desglose cuenta `notna()` por columna. El modelo clasifica el **instante de tiempo** como fuera de rango, no el sensor específico. Para identificar qué sensor concreto disparó la anomalía habría que añadir lógica adicional — mejora pendiente
- **`ensayo`: 69 puntos fuera de rango** — el modelo detecta días de ensayo como anomalía. Confirma que sin `ensayo` como feature el modelo no distingue experimento de anomalía real → justifica la mejora futura v4
- **Sensor Atascado: 433** — más de lo esperado. UVENT genera falsos positivos (ventana parada ≠ sensor atascado) → justifica exclusión de UVENT en v3

---

## Trabajo futuro — Líneas de mejora identificadas

### v4 — Incorporar columna `ensayo` como contexto del modelo

La columna `ensayo` del dataset combined registra los días de experimentos controlados (`ventilacion`, `calefaccion`, `deshumidificacion`, `co2`). En v2/v3 se filtra antes del modelo para evitar errores de tipo, pero su información es valiosa.

**Dos opciones para v4:**

| Opción | Qué hace | Ventaja |
|---|---|---|
| **A — Excluir días de ensayo del entrenamiento** | El modelo aprende solo operación normal | Definición de "normal" más limpia |
| **B — `ensayo` como feature (one-hot encoding)** | El modelo sabe que hay un ensayo activo | Reduce falsos positivos en días de experimento |

La Opción B es la más potente: el modelo aprendería que `XTINV=50°C` con `ensayo_calefaccion=1` es normal, pero con `ensayo_calefaccion=0` es anomalía. Requiere codificar `ensayo` con one-hot encoding antes de pasarlo al RF.

**No se implementa en v2/v3** para no alterar el baseline de comparación. Se propone como mejora en versión de producción o publicación futura.
