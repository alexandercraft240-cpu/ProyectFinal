# 🌸 Iris Flower Classifier — Proyecto Final de Analítica de Datos

Aplicación web completa para clasificar flores del dataset Iris usando
Regresión Logística y Red Neuronal (MLP) con una API Flask y frontend HTML.

---

## 📁 Estructura del Proyecto

```
/proyecto
│
├── train_models.py     ← Entrenamiento de modelos ML (ejecutar primero)
├── app.py              ← API REST con Flask (ejecutar segundo)
├── index.html          ← Frontend web (abrir en navegador)
├── requirements.txt    ← Dependencias de Python
├── README.md           ← Este archivo
│
├── /img                ← Imágenes de las flores (CREAR esta carpeta)
│   ├── setosa.jpg
│   ├── versicolor.jpg
│   └── virginica.jpg
│
│ (Estos archivos se generan automáticamente al ejecutar train_models.py)
├── scaler.pkl          ← Escalador de datos guardado
├── lr_model.pkl        ← Modelo de Regresión Logística guardado
├── nn_model.pkl        ← Red Neuronal guardada
└── metrics.json        ← Métricas de evaluación calculadas
```

---

## 🚀 Instalación y Ejecución

### Paso 1: Instalar dependencias

```bash
pip install -r requirements.txt
```

### Paso 2: Entrenar los modelos

```bash
python train_models.py
```

**Esto generará:**
- `scaler.pkl` — El escalador StandardScaler
- `lr_model.pkl` — Modelo de Regresión Logística
- `nn_model.pkl` — Red Neuronal (MLP)
- `metrics.json` — Accuracy y matrices de confusión

### Paso 3: Iniciar la API Flask

```bash
python app.py
```

El servidor estará en: `http://localhost:5000`

### Paso 4: Abrir el frontend

Abre `index.html` en tu navegador (doble clic o arrastra al navegador).

### Paso 5 (Opcional): Agregar imágenes

Crea la carpeta `img/` y agrega:
- `img/setosa.jpg`
- `img/versicolor.jpg`  
- `img/virginica.jpg`

Si no están, el frontend muestra emojis de fallback automáticamente.

---

## 🔌 Endpoints de la API

| Método | Ruta | Descripción |
|--------|------|-------------|
| POST | `/predict` | Predicción individual (1 flor) |
| POST | `/predict_batch` | Predicción por lotes (N flores) |
| GET | `/metrics` | Accuracy y matrices de confusión |
| GET | `/health` | Estado del servidor |

### Ejemplo: POST /predict

**Request:**
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Response:**
```json
{
  "logistic_regression": {
    "class": "setosa",
    "class_id": 0,
    "probabilities": {
      "setosa": 98.5,
      "versicolor": 1.2,
      "virginica": 0.3
    }
  },
  "neural_network": {
    "class": "setosa",
    "class_id": 0,
    "probabilities": { ... }
  },
  "input_features": { ... }
}
```

### Ejemplo: POST /predict_batch

**Request:**
```json
{
  "data": [
    [5.1, 3.5, 1.4, 0.2],
    [6.0, 2.7, 5.1, 1.6],
    [6.7, 3.3, 5.7, 2.5]
  ]
}
```

---

## 📊 Dataset Iris

| Variable | Descripción | Rango típico |
|----------|-------------|--------------|
| sepal_length | Largo del sépalo (cm) | 4.3 – 7.9 |
| sepal_width | Ancho del sépalo (cm) | 2.0 – 4.4 |
| petal_length | Largo del pétalo (cm) | 1.0 – 6.9 |
| petal_width | Ancho del pétalo (cm) | 0.1 – 2.5 |

**Clases:**
- `0` → Iris Setosa (pétalos pequeños, fácil de clasificar)
- `1` → Iris Versicolor (intermedia)
- `2` → Iris Virginica (pétalos grandes)

---

## 🤖 Modelos de Machine Learning

### Regresión Logística
- **¿Qué hace?** Calcula la probabilidad de pertenecer a cada clase
- **¿Por qué usarla?** Rápida, interpretable, buena baseline
- **Parámetros:** `max_iter=200, random_state=42`

### Red Neuronal (MLPClassifier)
- **Arquitectura:** 4 → 10 → 5 → 3 neuronas
- **Activación:** ReLU
- **¿Por qué usarla?** Captura patrones no lineales
- **Parámetros:** `hidden_layer_sizes=(10,5), activation='relu', max_iter=500`

---

## 🎓 Tecnologías Usadas

| Capa | Tecnología | Propósito |
|------|-----------|-----------|
| ML | Scikit-learn | Modelos y preprocesamiento |
| Datos | Pandas + NumPy | Manipulación de datos |
| Backend | Flask + Flask-CORS | API REST |
| Frontend | HTML + CSS + JS | Interfaz de usuario |
| Gráficas | Chart.js | Visualización de métricas |
| Fuentes | Google Fonts | Tipografía |

---

## ⚠️ Solución de Problemas

**La API no responde:**
- Verifica que Flask esté corriendo: `python app.py`
- Verifica el puerto 5000 no esté ocupado

**Error al cargar modelos:**
- Ejecuta primero: `python train_models.py`
- Verifica que `scaler.pkl`, `lr_model.pkl`, `nn_model.pkl` existan

**CORS error en el navegador:**
- Asegúrate que `flask-cors` esté instalado: `pip install flask-cors`

**Las imágenes no cargan:**
- Crea la carpeta `img/` con las imágenes correspondientes
- El frontend muestra emojis de fallback si no las encuentra
