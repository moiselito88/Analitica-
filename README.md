# 🧠 ObesityAI — Clasificador de Obesidad con Machine Learning

Sistema completo de clasificación de obesidad que combina un notebook de entrenamiento en Google Colab con una página web interactiva para predicciones en tiempo real.

---

## 📋 Descripción general

Este proyecto entrena dos modelos de Machine Learning sobre el dataset de obesidad sintético para predecir el nivel de obesidad de una persona a partir de sus características físicas y hábitos de vida. Los modelos entrenados se exportan como archivos JSON y son cargados por una página web que permite hacer predicciones sin necesitar Python ni servidores.

---

## 🗂️ Estructura del proyecto

```
📁 proyecto/
├── 📓 Modelo_final_documentado.ipynb      ← Notebook de entrenamiento (Google Colab)
├── 🌐 predictor_pro.html                  ← Página web de predicciones
├── 📄 metadata.json                       ← Clases, features, métricas y matrices
├── 📄 modelo_logistic_regression.json     ← Pesos de Regresión Logística
└── 📄 modelo_neural_network.json          ← Pesos de la Red Neuronal
```

> **Importante:** Los 3 archivos JSON deben estar en la **misma carpeta** que `predictor_pro.html` para que la página los cargue correctamente.

---

## 📊 Dataset

| Característica | Valor |
|---|---|
| **Archivo** | `ObesityDataSet_raw_and_data_sinthetic.csv` |
| **Total de registros** | 2 111 |
| **Entrenamiento** | 1 477 (70 %) |
| **Prueba** | 634 (30 %) |
| **Features de entrada** | 16 |
| **Clases objetivo** | 7 |

### Variables de entrada (16 features)

| Feature | Tipo | Descripción | Valores |
|---|---|---|---|
| `Gender` | Categórico | Sexo biológico | `Female: 0`, `Male: 1` |
| `Age` | Numérico | Edad en años | — |
| `Height` | Numérico | Estatura en metros | — |
| `Weight` | Numérico | Peso en kilogramos | — |
| `family_history_with_overweight` | Categórico | Antecedentes familiares de sobrepeso | `no: 0`, `yes: 1` |
| `FAVC` | Categórico | Consume alimentos hipercalóricos frecuentemente | `no: 0`, `yes: 1` |
| `FCVC` | Numérico | Frecuencia de consumo de verduras (1–3) | — |
| `NCP` | Numérico | Número de comidas principales al día | — |
| `CAEC` | Categórico | Consumo de alimentos entre comidas | `Always: 0`, `Frequently: 1`, `Sometimes: 2`, `no: 3` |
| `SMOKE` | Categórico | Fuma actualmente | `no: 0`, `yes: 1` |
| `CH2O` | Numérico | Consumo diario de agua en litros (1–3) | — |
| `SCC` | Categórico | Monitorea calorías consumidas | `no: 0`, `yes: 1` |
| `FAF` | Numérico | Frecuencia de actividad física (días/semana, 0–3) | — |
| `TUE` | Numérico | Tiempo de uso de pantallas (horas/día, 0–24) | — |
| `CALC` | Categórico | Frecuencia de consumo de alcohol | `Always: 0`, `Frequently: 1`, `Sometimes: 2`, `no: 3` |
| `MTRANS` | Categórico | Medio de transporte habitual | `Automobile: 0`, `Bike: 1`, `Motorbike: 2`, `Public_Transportation: 3`, `Walking: 4` |

### Clases objetivo (7 niveles)

| Índice | Clase | Descripción |
|---|---|---|
| 0 | `Insufficient_Weight` | Peso insuficiente |
| 1 | `Normal_Weight` | Peso normal |
| 2 | `Obesity_Type_I` | Obesidad tipo I |
| 3 | `Obesity_Type_II` | Obesidad tipo II |
| 4 | `Obesity_Type_III` | Obesidad tipo III |
| 5 | `Overweight_Level_I` | Sobrepeso nivel I |
| 6 | `Overweight_Level_II` | Sobrepeso nivel II |

---

## 🤖 Modelos

### Modelo 1 — Regresión Logística

Clasificador lineal que estima probabilidades por clase usando la función softmax sobre los logits.

**Arquitectura:**
```
Entrada (16 features) → StandardScaler
  → logits = coef(7×16) · x_norm + intercept(7)
  → softmax(logits) → probabilidades(7) → argmax → clase
```

**Parámetros guardados en `modelo_logistic_regression.json`:**

| Campo | Dimensión | Descripción |
|---|---|---|
| `coefficients` | 7 × 16 | Pesos aprendidos (uno por clase y feature) |
| `intercept` | 7 | Bias (sesgo) por clase |
| `scaler.mean` | 16 | Media de cada feature en el entrenamiento |
| `scaler.scale` | 16 | Desviación estándar de cada feature |

**Hiperparámetros ajustados con `RandomizedSearchCV` (10 iteraciones, cv=5):**

| Hiperparámetro | Valores probados |
|---|---|
| `C` (regularización) | 0.001 · 0.01 · 0.1 · 1 · 10 |
| `solver` | lbfgs · liblinear |
| `max_iter` | 500 · 1000 |

---

### Modelo 2 — Red Neuronal Artificial (MLP)

Perceptrón Multicapa con 3 capas ocultas y función de activación `tanh`. Captura relaciones no lineales que la Regresión Logística no puede modelar.

**Arquitectura:**
```
Entrada (16) → [128 neuronas, tanh] → [64 neuronas, tanh] → [32 neuronas, tanh] → Salida (7, softmax)
```

**Parámetros guardados en `modelo_neural_network.json`:**

| Capa | Pesos | Bias |
|---|---|---|
| Entrada → Capa 1 | 16 × 128 | 128 |
| Capa 1 → Capa 2 | 128 × 64 | 64 |
| Capa 2 → Capa 3 | 64 × 32 | 32 |
| Capa 3 → Salida | 32 × 7 | 7 |

- **Función de activación (capas ocultas):** `tanh`
- **Función de activación (capa de salida):** `softmax`
- **Total de parámetros:** 16×128 + 128 + 128×64 + 64 + 64×32 + 32 + 32×7 + 7 = **12 391**

**Hiperparámetros ajustados con `RandomizedSearchCV` (15 iteraciones, cv=5):**

| Hiperparámetro | Valores probados |
|---|---|
| `hidden_layer_sizes` | (64,) · (128,) · (64,32) · (128,64) · (128,64,32) |
| `activation` | relu · tanh |
| `alpha` (regularización L2) | 0.0001 · 0.001 · 0.01 |
| `learning_rate` | constant · adaptive |
| `early_stopping` | True |
| `validation_fraction` | 0.1 |

---

## 📈 Resultados

### Métricas sobre el conjunto de prueba (634 muestras)

| Métrica | Regresión Logística | Red Neuronal | Diferencia |
|---|---|---|---|
| **Accuracy** | **92.59 %** | **92.90 %** | +0.31 pp |
| **Precision** | 92.64 % | 93.00 % | +0.36 pp |
| **Recall** | 92.59 % | 92.90 % | +0.31 pp |
| **F1-Score** | 0.9257 | 0.9293 | +0.0036 |

> Ambos modelos superan el **92 %** de accuracy. La Red Neuronal es ligeramente superior gracias a su capacidad de capturar patrones no lineales.

### Matrices de confusión

**Regresión Logística** — 587 / 634 correctas

|  | Insuf. | Normal | OB-I | OB-II | OB-III | OW-I | OW-II |
|---|---|---|---|---|---|---|---|
| **Insuf.** | **82** | 0 | 0 | 0 | 0 | 0 | 0 |
| **Normal** | 5 | **73** | 0 | 0 | 0 | 8 | 0 |
| **OB-I** | 0 | 0 | **97** | 2 | 0 | 0 | 7 |
| **OB-II** | 0 | 0 | 1 | **88** | 0 | 0 | 0 |
| **OB-III** | 0 | 0 | 0 | 1 | **96** | 0 | 0 |
| **OW-I** | 0 | 8 | 0 | 0 | 0 | **74** | 5 |
| **OW-II** | 0 | 0 | 2 | 1 | 0 | 7 | **77** |

**Red Neuronal** — 589 / 634 correctas

|  | Insuf. | Normal | OB-I | OB-II | OB-III | OW-I | OW-II |
|---|---|---|---|---|---|---|---|
| **Insuf.** | **78** | 4 | 0 | 0 | 0 | 0 | 0 |
| **Normal** | 4 | **72** | 0 | 0 | 0 | 9 | 1 |
| **OB-I** | 0 | 0 | **100** | 1 | 0 | 0 | 5 |
| **OB-II** | 0 | 0 | 1 | **88** | 0 | 0 | 0 |
| **OB-III** | 0 | 0 | 0 | 1 | **96** | 0 | 0 |
| **OW-I** | 0 | 8 | 0 | 0 | 0 | **76** | 3 |
| **OW-II** | 0 | 0 | 1 | 0 | 0 | 7 | **79** |

> La mayor fuente de errores en ambos modelos se concentra en la confusión entre clases adyacentes (Normal ↔ Overweight, OB-I ↔ OB-II), lo cual es esperable dado que estas categorías comparten límites borrosos.

---

## ⚙️ Pipeline de entrenamiento

```
1. Cargar CSV (2111 filas × 17 columnas)
        ↓
2. LabelEncoder → convertir 8 columnas categóricas a enteros
        ↓
3. Separar X (16 features) e y (7 clases codificadas)
        ↓
4. train_test_split (70% train / 30% test, stratify=y, random_state=42)
        ↓
5. StandardScaler → z = (x − μ) / σ  [fit solo en train, transform en ambos]
        ↓
        ┌──────────────────────┬──────────────────────┐
        │  Regresión Logística │   Red Neuronal (MLP) │
        │  RandomSearchCV      │   RandomSearchCV     │
        │  10 iter × cv=5      │   15 iter × cv=5     │
        │  → best_estimator_   │   → best_estimator_  │
        └──────────────────────┴──────────────────────┘
        ↓
6. Evaluar sobre test → accuracy, precision, recall, F1, matriz de confusión
        ↓
7. Exportar a 3 archivos JSON (metadata + lr + nn)
        ↓
8. Descargar desde Colab → colocar en carpeta del HTML
```

---

## 🚀 Cómo usar

### Paso 1 — Entrenar en Google Colab

1. Abre `Modelo_final_documentado.ipynb` en [Google Colab](https://colab.research.google.com)
2. Ejecuta todas las celdas en orden (`Ctrl + F9` → Ejecutar todo)
3. Cuando lo pida, sube el archivo `ObesityDataSet_raw_and_data_sinthetic.csv`
4. Descarga los 3 JSON que se generan al final:
   - `metadata.json`
   - `modelo_logistic_regression.json`
   - `modelo_neural_network.json`

### Paso 2 — Configurar la página web

Coloca los 4 archivos en la misma carpeta:

```
📁 mi-carpeta/
├── predictor_pro.html
├── metadata.json
├── modelo_logistic_regression.json
└── modelo_neural_network.json
```

### Paso 3 — Abrir en el navegador

Abre `predictor_pro.html` directamente en el navegador (doble clic).  
La barra superior mostrará **"Modelos cargados ✓"** cuando todo esté listo.

---

## 🌐 Funcionalidades de la página web

### Panel izquierdo — Predicción Individual
- Elige el modelo (Regresión Logística o Red Neuronal)
- Los **8 campos categóricos** tienen desplegables con los valores reales del dataset
- Los **8 campos numéricos** aceptan valores con su rango correspondiente
- El resultado muestra la clase predicha y las **barras de probabilidad** para las 7 clases ordenadas de mayor a menor confianza

### Panel derecho — Predicción por Lotes
- Sube cualquier CSV con los mismos encabezados del dataset
- Si el CSV contiene la columna `NObeyesdad`, las **métricas se calculan con los datos reales**
- **Pestaña Métricas:** Accuracy, Precision, Recall, F1-Score
- **Pestaña Matriz:** Matriz de confusión 7×7 con exactitud calculada, aciertos y errores totales
- **Pestaña Predicciones:** Top 10 por confianza + botón "Ver todos" + descarga CSV

---

## 🔧 Implementación en JavaScript (predicción sin Python)

La página web implementa los modelos directamente en JS:

**Normalización (StandardScaler):**
```js
x_norm[i] = (x[i] - scaler_mean[i]) / scaler_scale[i]
```

**Regresión Logística:**
```js
logits[c] = dot(coefficients[c], x_norm) + intercept[c]
probs = softmax(logits)                    // suma = 1
clase = argmax(probs)
```

**Red Neuronal (forward pass):**
```js
a1 = tanh(W1 · x_norm + b1)   // 16 → 128
a2 = tanh(W2 · a1     + b2)   // 128 → 64
a3 = tanh(W3 · a2     + b3)   // 64 → 32
probs = softmax(W4 · a3 + b4)  // 32 → 7
clase = argmax(probs)
```

---

## 📦 Dependencias

### Notebook
| Librería | Uso |
|---|---|
| `scikit-learn` | Modelos, preprocesamiento, métricas, búsqueda de hiperparámetros |
| `pandas` | Carga y manipulación del dataset |
| `numpy` | Operaciones matriciales y serialización a JSON |
| `matplotlib` | Visualización de figuras y matrices |
| `seaborn` | Heatmaps de matrices de confusión |

### Página web
| Recurso | Uso |
|---|---|
| Google Fonts (Syne, Outfit, JetBrains Mono) | Tipografía |
| Navegador moderno (Chrome / Firefox / Safari / Edge) | `fetch()`, ES6+, CSS Variables |

> La página **no requiere servidor, instalación ni conexión a internet** una vez descargados los archivos.

---

## 👤 Autor

Desarrollado como proyecto de Analítica de Datos  
Universidad Cooperativa de Colombia · 2026
