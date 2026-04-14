# Clasificación y Corrección de Anomalías en Sensores de Invernadero

## Descripción

Proyecto de **Trabajo Fin de Máster (TFM)** para la detección y corrección automática de anomalías en datos de sensores de invernadero. El sistema:

1. Define y sintetiza 6 tipos de anomalías sobre datos reales de sensores
2. Entrena un modelo ML para **detectar** si un registro es anómalo (binario)
3. Entrena un segundo modelo ML para **clasificar el tipo** de anomalía (multi-clase)
4. Aplica métodos de **corrección** específicos por tipo de anomalía

---

## Estado del Proyecto

| Componente | Estado |
|---|---|
| Notebook activo | `Ultimo antes del cambio Clasificación_y_Corrección_de_Anomalías.ipynb` |
| Dataset CSV requerido | **AUSENTE** — no incluido en el repositorio |
| Fichero de estado `.pkl` | Se genera al ejecutar (3 checkpoints intermedios) |
| `requirements.txt` | **AUSENTE** — instalación manual requerida |
| Modelos entrenados exportados | Se guardan al ejecutar (`modelo_1_detector.joblib`, `modelo_2_clasificador.joblib`) |

> **IMPORTANTE:** El proyecto **no puede ejecutarse** sin el fichero CSV de datos. Ver sección [Datos Requeridos](#datos-requeridos).

---

## Estructura del Repositorio

```
Clasificaci-n-y-correcci-n/
├── README.md
├── Ultimo antes del cambio Clasificación_y_Corrección_de_Anomalías.ipynb  ← Notebook activo (448 KB)
└── Versión TFM Clasificación_y_Corrección_de_Anomalías.ipynb              ← Versión TFM (referencia, no usar)
```

### Ficheros que se generan al ejecutar

```
├── imputador_nans.joblib              ← Imputer ajustado
├── modelo_1_detector.joblib           ← Random Forest detección
├── modelo_2_clasificador.joblib       ← Random Forest clasificación de tipo
├── features_modelo_1.joblib           ← Lista de features usadas
├── label_encoder_modelo_2.joblib      ← Encoder de etiquetas de tipo
└── estado_completo_proceso.pkl        ← Checkpoints intermedios (3 puntos)
```

### Ficheros que deben añadirse para ejecución

```
├── 2020_10_10-2020_12_29.csv          ← Dataset greenhouse (REQUERIDO)
└── requirements.txt                   ← Dependencias (ver abajo)
```

---

## Datos Requeridos

### Fichero: `2020_10_10-2020_12_29.csv`

| Propiedad | Valor |
|---|---|
| Nombre exacto | `2020_10_10-2020_12_29.csv` |
| Registros | 233,381 filas |
| Columnas | 13 |
| Período | 10/10/2020 al 29/12/2020 |
| Frecuencia de muestreo | ~5 minutos |
| Formato de fecha | `%d/%m/%Y %H:%M:%S` |

### Columnas del CSV

| Variable | Descripción | Sensor / Modelo | Rango físico real | Precisión |
|---|---|---|---|---|
| `Fecha` | Timestamp del registro | — | Dic 2023 → Dic 2024 | — |
| `PCO2EXT` | CO2 exterior (ppm) | E+E Elektronik EE820-HV1A6E1 | 0 – 2000 ppm | ±(50 ppm + 2%) |
| `PHEXT` | Humedad relativa exterior (%) | Campbell HC2A-S3 | 0 – 100 % | ±0.8 % rH |
| `PRAD` | Radiación global exterior (W/m²) | Campbell SP-214-SS | 0 – 2000 W/m² | — |
| `PRGINT` | Radiación global interior (W/m²) | Campbell SP-110-SS | 0 – 2000 W/m² | ±1 % |
| `PTEXT` | Temperatura exterior (°C) | Campbell HC2A-S3 | -40 – 60 °C | ±0.1 °C |
| `PVV` | Velocidad del viento (m/s) | Wittich & Visser PA2 | 0 – 45.8 m/s | 0.5 m/s |
| `UVENT_cen` | Apertura media ventanas centrales (%) | De Gier I-DE (encoder) | 0 – 100 % | lineal |
| `UVENT_lN` | Apertura media ventanas laterales (%) | De Gier I-DE (encoder) | 0 – 100 % | lineal |
| `XCO2I` | CO2 interior (ppm) | E+E Elektronik EE820-HV1A6E1 | 0 – 2000 ppm | ±(50 ppm + 2%) |
| `XHINV` | Humedad relativa interior (%) | Campbell HC2A-S3 | 0 – 100 % | ±0.8 % rH |
| `XTINV` | Temperatura interior (°C) | Campbell HC2A-S3 | -40 – 60 °C | ±0.1 °C |
| `XTS` | Temperatura suelo 5 cm (°C) | Campbell 109 (NTC) | -50 – 70 °C | ±0.2 °C |

> Los rangos físicos reales provienen de las fichas técnicas del sistema AGROCONNECT (PLC1 y PLC6).
> Fuente: `Dataset/Metadatos_SensoresyActuadores/AGROCONNECT_Variables_PLCs.xlsx`

#### Ventanas de ventilación — agregación

`UVENT_cen` es la media de las posiciones reales de 6 ventanas cenitales (UVCEN1_1, UVCEN1_2, UVCEN1_3, UVCEN2_1, UVCEN2_2, UVCEN2_3).
`UVENT_lN` es la media de las posiciones reales de 7 ventanas laterales (UVLAT1N, UVLAT1ON, UVLAT1OS, UVLAT1S, UVLAT2E, UVLAT2N, UVLAT2S).
Datos de ventilación disponibles **desde el 5 de marzo de 2024** (~27 % del dataset total con NaN antes de esa fecha).

El dataset no debe contener valores nulos en el original — los `NaN` se introducen sintéticamente durante la ejecución, excepto los NaN de ventilación que son estructurales (sensor no instalado en el período inicial).

---

## Instalación y Configuración

### Requisitos del sistema

- Python 3.11.7
- macOS (probado) / Linux
- ~4 GB RAM disponible para Random Forest sobre 233K registros

### Crear entorno virtual

```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate   # Windows
```

### Instalar dependencias

```bash
pip install -U pip
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
# Solo en Mac con GPU Apple Silicon:
pip install torch torchvision torchaudio
```

### `requirements.txt` recomendado

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
tensorflow>=2.13
joblib>=1.3
```

---

## Cómo Ejecutar

### Prerequisitos

1. Entorno virtual activado (`.venv`)
2. Dataset CSV colocado en la raíz del proyecto con el nombre exacto: `2020_10_10-2020_12_29.csv`
3. Jupyter Notebook o JupyterLab instalado

### Lanzar el notebook

```bash
source .venv/bin/activate
jupyter notebook "Versión TFM Clasificación_y_Corrección_de_Anomalías.ipynb"
```

O con JupyterLab:

```bash
jupyter lab
```

### Orden de ejecución

Ejecutar las celdas **en orden secuencial** de arriba a abajo. Las etapas son:

| Etapa | Celdas | Descripción |
|---|---|---|
| 1. Setup | Inicio | Importaciones, carga del CSV |
| 2. Exploración | — | Estadísticas, matriz de correlación |
| 3. Inyección de anomalías | — | Generación sintética de los 6 tipos |
| 4. Feature Engineering | — | Columnas indicadoras de NaN |
| 5. Modelo 1 | — | Detección Normal vs. Anomalía |
| 6. Modelo 2 | — | Clasificación del tipo de anomalía |
| 7. Corrección | — | Métodos de corrección por tipo |
| 8. Evaluación | — | RMSE, MAE, análisis de diferencias |

> El notebook tiene 115 celdas. La ejecución completa tarda varios minutos (entrenamiento de Random Forest sobre ~233K registros).

---

## Pipeline del Sistema

```
CSV (233,381 registros × 13 columnas)
        │
        ▼
Extracción de features temporales (Hora, DiaSemana, Mes)
        │
        ▼
Inyección sintética de anomalías (19,981 filas ≈ 8.56%)
  ├─ Datos Faltantes       →    538 filas
  ├─ Sensor Atascado       →  1,958 filas
  ├─ Ruido / Spikes        →  6,802 filas
  ├─ Fuera de Rango        →  4,399 filas
  ├─ Desviación Correlación → 1,617 filas
  └─ Contextual            →    538 filas
        │
        ▼
Modelo 1: Detección (Random Forest binario)
  → Accuracy: 99.66%  |  AUC: 0.9999
        │
        ▼
Modelo 2: Clasificación de tipo (Random Forest multi-clase)
  → Accuracy: 88.69%
        │
        ▼
Corrección por tipo
  ├─ Datos Faltantes   → IterativeImputer     (RMSE: 0.733)
  ├─ Sensor Atascado   → Híbrido estacional   (RMSE: 1.806–14.708)
  ├─ Ruido             → Vecinos locales      (RMSE: 22.373)
  ├─ Fuera de Rango    → Vecinos locales      (RMSE: 40.038)
  ├─ Desviación Corr.  → Vecinos locales
  └─ Contextual        → Vecinos locales      (RMSE: 0.448)
        │
        ▼
Evaluación: diferencias con dataset original
```

---

## Tipos de Anomalías

### 1. Datos Faltantes (`datos_faltantes`)
Valores `NaN` en uno o más sensores. Causa: fallo de comunicación, sensor desconectado.

**Inyección:** 2% de filas, 1 sensor aleatorio por fila.
**Corrección:** `IterativeImputer` con Random Forest regressor.
**RMSE corrección:** 0.733

### 2. Sensor Atascado (`sensor_atascado`)
El sensor devuelve el mismo valor durante un período prolongado. Causa: congelamiento del hardware, fallo de firmware.

**Inyección:** 150 secuencias, duración 5–20 intervalos.
**Corrección:** Promedio estacional (sensores dinámicos) o interpolación lineal (sensores estables).
**RMSE corrección:** 1.806 (estables) / 14.708 (dinámicos)

### 3. Ruido / Spikes (`ruido`)
Fluctuaciones erráticas que se desvían drásticamente de la tendencia. Causa: interferencias eléctricas, errores de ADC.

**Inyección:** 3% de filas, magnitud 3–6× desviación estándar.
**Corrección:** Promedio de vecinos inmediatos con umbral 2×std.
**RMSE corrección:** 22.373

### 4. Valores Fuera de Rango (`fuera_de_rango`)
Lecturas que exceden los percentiles P0.1/P99.9 calculados del propio dataset. Causa: saturación del sensor, fallo electrónico.

**Definición de rango:** estadística — percentil 0.1 y 99.9 del dataset original por cada sensor (no rangos físicos fijos).
**Inyección:** 2% de filas, valores un 10% más allá de esos percentiles.
**Corrección:** Promedio de vecinos inmediatos con umbral 2×std.
**RMSE corrección:** 40.038

> **Limitación:** los rangos son globales, no estacionales. Un valor anómalo solo para una época del año pero dentro del percentil global no se detecta como "Fuera de Rango".

### 5. Desviación de Correlación (`desviacion_correlacion`)
Un sensor rompe la correlación esperada con sensores relacionados. Causa: fallo de calibración, obstrucción física.

**Inyección:** 3% de filas en pares de alta correlación (PRAD–PRGINT, XTINV–XHINV).
**Corrección:** Promedio de vecinos con umbral 2×std.

### 6. Contextual (`contextual`)
Valores normales en aislamiento pero anómalos en su contexto. Causa: lógica de sistema errónea, actuadores defectuosos.

**Variantes implementadas:**
- *Iluminación interior nocturna:* PRGINT > 0 cuando PRAD ≈ 0 (de noche). → 538 filas.
- *Falta de respuesta CO2 a ventilación:* XCO2I constante con ventilación alta. → **0 filas generadas** (ver Problemas Conocidos).

**Corrección:** Promedio de vecinos con umbral 2×std.
**RMSE corrección:** 0.448

---

## Resultados de los Modelos

### Modelo 1: Detección de Anomalías

| Métrica | Valor |
|---|---|
| Accuracy | **99.66%** |
| ROC AUC | **0.9999** |
| Average Precision | **0.9992** |
| Falsos Positivos | 18 |
| Falsos Negativos | 223 |

**Tasa de detección por tipo:**

| Tipo | Detección |
|---|---|
| Contextual | 100.0% |
| Datos Faltantes | 99.9% |
| Fuera de Rango | 99.8% |
| Desviación Correlación | 97.6% |
| Ruido | 94.9% |
| Sensor Atascado | **82.4%** ← punto débil |

### Modelo 2: Clasificación del Tipo

| Métrica global | Valor |
|---|---|
| Accuracy | **88.69%** |

| Tipo | Precisión | Recall |
|---|---|---|
| Contextual | 1.00 | 1.00 |
| Datos Faltantes | 1.00 | 1.00 |
| Desviación Correlación | 1.00 | 0.99 |
| Sensor Atascado | 1.00 | 0.99 |
| Ruido | **0.79** | 0.92 |
| Fuera de Rango | **0.84** | **0.61** ← punto débil |

---

## Problemas Conocidos y Limitaciones

### CRÍTICOS

#### 1. Dataset ausente
El fichero `2020_10_10-2020_12_29.csv` **no está incluido** en el repositorio. Sin él el notebook falla en la primera celda de carga de datos.

#### 2. Fichero de estado PKL ausente
El código referencia `estado_completo_proceso.pkl` para guardar checkpoints entre secciones. No existe aún (se genera durante la primera ejecución completa).

---

### DETECCIÓN

#### 3. Sensor Atascado: tasa de detección del 82.4%
**Causa:** Un sensor atascado en un valor plausible es indistinguible de un sensor estable sin contexto temporal explícito en las features. El modelo no tiene features de ventana temporal (rolling mean, diff) que capturen la repetición.

**Solución propuesta:** Añadir features como `rolling_std(window=10)`, `diff_lag1`, y `runs_of_equal_values` por sensor.

#### 4. Confusión entre Ruido y Fuera de Rango
**Causa:** Ambos tipos generan desvíos bruscos de valor. Sin features que capturen la magnitud relativa al rango físico del sensor vs. la variabilidad local, el modelo los confunde.

- Ruido: desviación grande respecto a vecinos locales pero dentro del rango físico
- Fuera de Rango: valor fuera del límite físico absoluto

**Solución propuesta:** Añadir features `(valor - rango_min) / (rango_max - rango_min)` (posición relativa en el rango físico) y `z_score_local(window=20)`.

---

### CORRECCIÓN

#### 5. Corrección introduce más cambios que las anomalías originales
Se observa que tras la corrección hay **más diferencias** con el dataset original que filas anómalas inyectadas. Ejemplo: PCO2EXT tenía 1,517 anomalías pero 4,353 diferencias post-corrección.

**Causa:** Los métodos de vecinos locales con umbral fijo (2×std) modifican también valores que son normales pero con variabilidad legítima alta.

**Solución propuesta:** Aplicar corrección solo cuando la predicción del modelo tenga alta confianza (probabilidad > 0.85), y usar umbrales adaptativos por sensor.

#### 6. RMSE alto en Ruido (22.373) y Fuera de Rango (40.038)
**Causa:** El método de vecinos inmediatos no captura bien las dinámicas de sensores como PRAD (radiación solar) que tienen transiciones rápidas legítimas.

**Solución propuesta:** Para sensores con alta variabilidad (PRAD, PRGINT), usar reconstrucción por modelo físico (correlación PRAD↔PRGINT) en lugar de promedio simple de vecinos.

#### 7. Anomalía contextual CO2 — 0 secuencias generadas
**Causa:** La condición "alta ventilación + CO2 exterior bajo + CO2 interior alto" ocurre menos del 0.01% del tiempo en el dataset. Con solo 1% de inyección objetivo y 100 intentos, no se encuentra ninguna ventana válida.

**Solución propuesta:** Relajar la condición de ventilación (umbral ≥ 30% en lugar de ≥ 50%) o usar simulación Monte Carlo sobre segmentos de datos que cumplan condiciones parciales.

---

### DISEÑO

#### 8. Thresholds hardcodeados
El umbral de corrección `2×std_dev` es igual para todos los sensores. Sensores con distribuciones asimétricas (PRAD, PVV) requieren umbrales asimétricos.

#### 9. Sin validación cruzada temporal
El split 80/20 es aleatorio, no respeta el orden temporal. En series temporales esto causa data leakage (el modelo puede ver el "futuro" del pasado). Usar `TimeSeriesSplit` de scikit-learn.

---

## Mejoras Recomendadas

### Corto plazo (para hacer ejecutable)
- [ ] Añadir `2020_10_10-2020_12_29.csv` al repositorio o documentar cómo obtenerlo
- [ ] Crear `requirements.txt` con versiones fijas

### Medio plazo (calidad del modelo)
- [ ] Añadir features temporales de ventana deslizante para mejorar detección de Sensor Atascado
- [ ] Añadir features de posición en rango físico para distinguir Ruido vs. Fuera de Rango
- [ ] Usar `TimeSeriesSplit` en lugar de split aleatorio
- [ ] Añadir umbrales de confianza mínima antes de aplicar corrección

### Largo plazo (producción)
- [ ] Modularizar en scripts Python (`src/detection.py`, `src/correction.py`, `src/models.py`)
- [ ] Añadir CLI con `argparse` para procesar nuevos CSVs
- [ ] Implementar logging con el módulo `logging`
- [ ] Añadir validación del esquema del CSV al cargarlo
- [ ] Tests unitarios para funciones de corrección

---

## Contacto y Referencia

Proyecto desarrollado como **Trabajo Fin de Máster** sobre detección y corrección de anomalías en datos de sensores de entornos controlados (invernaderos).

Dataset: AGROCONNECT — datos de sensores y actuadores de invernadero, período Octubre–Diciembre 2020.
