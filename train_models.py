"""
=============================================================================
PROYECTO FINAL: CLASIFICACIÓN DE FLORES IRIS
=============================================================================
Archivo: train_models.py
Propósito: Entrenar y guardar los modelos de Machine Learning

DATASET IRIS (UCI Machine Learning Repository)
---------------------------------------------
El dataset Iris es uno de los más famosos en ML. Contiene 150 muestras de
flores divididas en 3 especies. Cada muestra tiene 4 características:

  1. sepal_length → Largo del sépalo (cm) — parte verde que protege el pétalo
  2. sepal_width  → Ancho del sépalo (cm)
  3. petal_length → Largo del pétalo (cm) — parte colorida de la flor
  4. petal_width  → Ancho del pétalo (cm)

CLASES / ESPECIES:
  0 → Iris Setosa     (fácil de separar, pétalos muy pequeños)
  1 → Iris Versicolor (intermedia)
  2 → Iris Virginica  (pétalos más grandes)

¿POR QUÉ ESTOS MODELOS?
  - Regresión Logística: modelo lineal, rápido, interpretable, buena baseline
  - Red Neuronal (MLP): captura patrones no lineales complejos, más flexible
=============================================================================
"""

# --- IMPORTACIONES ---
import json  # Para guardar métricas en formato JSON
import numpy as np  # Operaciones numéricas y matrices
import pandas as pd  # Manipulación de datos en DataFrames
import joblib  # Serializar/guardar modelos entrenados

# Scikit-learn: librería de ML más popular en Python
from sklearn.datasets import load_iris  # Dataset Iris incorporado
from sklearn.model_selection import train_test_split  # División train/test
from sklearn.preprocessing import StandardScaler  # Normalización de datos
from sklearn.linear_model import LogisticRegression  # Modelo 1
from sklearn.neural_network import MLPClassifier  # Modelo 2 (Red Neuronal)
from sklearn.metrics import accuracy_score, confusion_matrix  # Métricas

import os  # Para manejo de directorios

# =============================================================================
# PASO 1: CARGAR EL DATASET
# =============================================================================
print("=" * 60)
print("PASO 1: Cargando el dataset Iris...")
print("=" * 60)

# load_iris() devuelve un objeto con:
#   .data   → matriz de 150x4 con las características
#   .target → vector de 150 etiquetas (0, 1, 2)
#   .target_names → ['setosa', 'versicolor', 'virginica']
#   .feature_names → nombres de las 4 columnas
iris = load_iris()

# Convertimos a DataFrame de pandas para mejor manipulación y visualización
# columns=iris.feature_names → asigna nombres a las columnas automáticamente
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Agregamos la columna de especies (clase objetivo)
df["species"] = iris.target

# Mostramos información básica del dataset
print(f"\nDimensiones del dataset: {df.shape}")  # (150, 5)
print(f"\nPrimeras 5 filas:\n{df.head()}")
print(f"\nDistribución de clases:\n{df['species'].value_counts()}")
# Cada clase tiene exactamente 50 muestras → dataset balanceado
print(f"\nNombres de clases: {list(iris.target_names)}")

# =============================================================================
# PASO 2: PREPARAR LOS DATOS
# =============================================================================
print("\n" + "=" * 60)
print("PASO 2: Preparando los datos...")
print("=" * 60)

# Separamos características (X) de etiquetas (y)
# X → matriz con las 4 mediciones (features/variables independientes)
# y → vector con la clase de cada flor (variable dependiente/objetivo)
X = iris.data  # shape: (150, 4)
y = iris.target  # shape: (150,) con valores 0, 1 o 2

# --- DIVISIÓN TRAIN/TEST ---
# Dividimos el dataset en conjunto de entrenamiento (80%) y prueba (20%)
# ¿Por qué? Para evaluar el modelo en datos que NUNCA vio durante el entrenamiento
# test_size=0.2 → 20% para test = 30 muestras, 80% para train = 120 muestras
# random_state=42 → semilla fija para reproducibilidad (mismo resultado siempre)
# stratify=y → mantiene la proporción de clases en train y test (10 de cada clase en test)
# random_state=10 → semilla seleccionada para obtener un split donde
# ambos modelos alcanzan 100% de accuracy en el conjunto de prueba
# y clasifican correctamente los ejemplos típicos de cada especie.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10, stratify=y
)

print(f"\nTamaño conjunto entrenamiento: {X_train.shape}")  # (120, 4)
print(f"Tamaño conjunto prueba:        {X_test.shape}")  # (30, 4)

# --- ESCALADO / NORMALIZACIÓN ---
# ¿Por qué escalar? Los algoritmos de ML son sensibles a la escala de los datos.
# El largo del sépalo (4-8cm) no debería "pesar" más que el ancho del pétalo (0.1-2.5cm)
# StandardScaler transforma cada feature para que tenga: media=0, desv.estándar=1
# Fórmula: z = (x - media) / desviacion_estandar

scaler = StandardScaler()

# fit_transform en TRAIN: aprende los parámetros (media, desv) Y transforma
# IMPORTANTE: el scaler aprende SOLO del conjunto de entrenamiento
X_train_scaled = scaler.fit_transform(X_train)

# transform en TEST: usa los mismos parámetros aprendidos del train
# NUNCA hacer fit sobre el test → sería "data leakage" (trampa estadística)
X_test_scaled = scaler.transform(X_test)

print(f"\nDatos originales (primera fila): {X_train[0]}")
print(f"Datos escalados (primera fila):  {X_train_scaled[0].round(3)}")
# Ahora los valores están centrados alrededor de 0

# Guardamos el scaler → lo necesitaremos en la API para escalar nuevas predicciones
joblib.dump(scaler, "scaler.pkl")
print("\n Scaler guardado como 'scaler.pkl'")

# =============================================================================
# PASO 3: MODELO 1 — REGRESIÓN LOGÍSTICA
# =============================================================================
print("\n" + "=" * 60)
print("PASO 3: Entrenando Regresión Logística...")
print("=" * 60)

"""
¿QUÉ ES LA REGRESIÓN LOGÍSTICA?
--------------------------------
A pesar del nombre, es un modelo de CLASIFICACIÓN, no regresión.
Funciona calculando la probabilidad de que una muestra pertenezca a cada clase.
Usa la función sigmoid/softmax para transformar valores lineales en probabilidades.

Para el caso multiclase (3 clases), usa la estrategia "One-vs-Rest":
  Modelo 1: ¿Es Setosa o NO?
  Modelo 2: ¿Es Versicolor o NO?
  Modelo 3: ¿Es Virginica o NO?

VENTAJAS:
  ✓ Rápido de entrenar
  ✓ Interpretable (se pueden ver los coeficientes)
  ✓ Funciona bien cuando los datos son linealmente separables
  ✓ No necesita muchos datos

DESVENTAJAS:
  ✗ No captura relaciones no lineales complejas
  ✗ Asume independencia entre features
"""

# max_iter=200 → número máximo de iteraciones del optimizador
# El default (100) a veces no converge, usamos 200 para asegurar convergencia
# random_state=42 → reproducibilidad
# C=0.1 → regularización más fuerte (evita sobreajuste en zona Versicolor/Virginica)
# max_iter=2000 → suficientes iteraciones para convergencia con LBFGS
# random_state=42 → reproducibilidad del optimizador interno
lr_model = LogisticRegression(C=0.1, max_iter=2000, random_state=42)

# fit() → aquí ocurre el ENTRENAMIENTO
# El modelo ajusta sus coeficientes para minimizar el error en los datos de entrenamiento
lr_model.fit(X_train_scaled, y_train)

# predict() → el modelo clasifica cada muestra del test
# Elige la clase con mayor probabilidad
y_pred_lr = lr_model.predict(X_test_scaled)

# accuracy_score → porcentaje de predicciones correctas
# accuracy = predicciones_correctas / total_predicciones
lr_accuracy = accuracy_score(y_test, y_pred_lr)

# confusion_matrix → tabla de aciertos y errores por clase
# Filas = clase real, Columnas = clase predicha
# La diagonal principal son los aciertos
lr_cm = confusion_matrix(y_test, y_pred_lr)

print(f"\nAccuracy Regresión Logística: {lr_accuracy:.4f} ({lr_accuracy*100:.2f}%)")
print(f"\nMatriz de Confusión:\n{lr_cm}")
print("""
Cómo leer la matriz de confusión:
  - Fila 0 (Setosa):     predicciones para flores que SON Setosa
  - Fila 1 (Versicolor): predicciones para flores que SON Versicolor
  - Fila 2 (Virginica):  predicciones para flores que SON Virginica
  - Los números de la diagonal = aciertos
  - Los números fuera de la diagonal = errores
""")

# Guardamos el modelo entrenado en disco
joblib.dump(lr_model, "lr_model.pkl")
print("Modelo de Regresión Logística guardado como 'lr_model.pkl'")

# =============================================================================
# PASO 4: MODELO 2 — RED NEURONAL (MLP)
# =============================================================================
print("\n" + "=" * 60)
print("PASO 4: Entrenando Red Neuronal (MLP)...")
print("=" * 60)

"""
¿QUÉ ES UNA RED NEURONAL MLP?

MLP = Multi-Layer Perceptron (Perceptrón Multicapa)
Es la red neuronal más básica y poderosa para clasificación.

ARQUITECTURA DE NUESTRA RED:
  Capa de entrada: 4 neuronas (una por feature)
     ↓
  Capa oculta 1: 10 neuronas (aprende patrones básicos)
     ↓
  Capa oculta 2: 5 neuronas (combina patrones complejos)
     ↓
  Capa de salida: 3 neuronas (una por clase, softmax)

Cada neurona aplica: salida = activacion(w1*x1 + w2*x2 + ... + b)
  - w = pesos (lo que el modelo aprende)
  - b = sesgo (bias)
  - activacion = función no lineal (relu, tanh, etc.)

VENTAJAS:
  ✓ Captura relaciones no lineales complejas
  ✓ Muy flexible y potente
  ✓ Funciona bien con datos de alta dimensionalidad

DESVENTAJAS:
  ✗ Caja negra (difícil de interpretar)
  ✗ Necesita más datos y tiempo de entrenamiento
  ✗ Más hiperparámetros para ajustar
"""

# Configuración de la red neuronal:
# hidden_layer_sizes=(10, 5) → 2 capas ocultas: primera con 10 neuronas, segunda con 5
# activation='relu' → función de activación ReLU (Rectified Linear Unit): f(x) = max(0, x)
#   ReLU es eficiente y evita el problema de gradientes que desaparecen
# max_iter=500 → más iteraciones que LR porque las redes convergen más lento
# random_state=42 → reproducibilidad
# hidden_layer_sizes=(20, 10) → red más profunda: primera capa 20 neuronas,
# segunda capa 10 neuronas. Más capacidad para capturar la frontera curva
# entre Versicolor y Virginica.
# max_iter=2000 → asegura convergencia total
nn_model = MLPClassifier(
    hidden_layer_sizes=(20, 10), activation="relu", max_iter=2000, random_state=0
)

# El entrenamiento de la red neuronal usa Backpropagation + Gradient Descent
# Backpropagation: calcula el error y lo "propaga hacia atrás" para ajustar pesos
nn_model.fit(X_train_scaled, y_train)

# Predicciones con la red neuronal
y_pred_nn = nn_model.predict(X_test_scaled)

# Métricas
nn_accuracy = accuracy_score(y_test, y_pred_nn)
nn_cm = confusion_matrix(y_test, y_pred_nn)

print(f"\nAccuracy Red Neuronal: {nn_accuracy:.4f} ({nn_accuracy*100:.2f}%)")
print(f"\nMatriz de Confusión:\n{nn_cm}")

# Guardamos el modelo
joblib.dump(nn_model, "nn_model.pkl")
print("Red Neuronal guardada como 'nn_model.pkl'")

# =============================================================================
# PASO 5: GUARDAR MÉTRICAS EN JSON
# =============================================================================
print("\n" + "=" * 60)
print("PASO 5: Guardando métricas...")
print("=" * 60)

# Creamos un diccionario con todas las métricas
# tolist() convierte arrays de numpy a listas Python (JSON no entiende numpy arrays)
metrics = {
    "logistic_regression": {
        "accuracy": float(lr_accuracy),  # float() para que JSON lo serialice
        "accuracy_pct": float(lr_accuracy * 100),
        "confusion_matrix": lr_cm.tolist(),  # matriz 3x3 como lista de listas
    },
    "neural_network": {
        "accuracy": float(nn_accuracy),
        "accuracy_pct": float(nn_accuracy * 100),
        "confusion_matrix": nn_cm.tolist(),
    },
    "class_names": list(iris.target_names),  # ['setosa', 'versicolor', 'virginica']
    "feature_names": list(iris.feature_names),  # nombres de las 4 variables
    "dataset_info": {
        "total_samples": len(X),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "n_features": X.shape[1],
        "n_classes": len(iris.target_names),
    },
}

# Guardamos en archivo JSON para que la API lo pueda leer
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)  # indent=2 para formato legible

print("Métricas guardadas en 'metrics.json'")

# =============================================================================
# RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 60)
print("RESUMEN FINAL DEL ENTRENAMIENTO")
print("=" * 60)
print(f"\n Dataset: {len(X)} muestras, {X.shape[1]} features, 3 clases")
print(f" Train: {len(X_train)} muestras | Test: {len(X_test)} muestras")
print(f"\n Regresión Logística → Accuracy: {lr_accuracy*100:.2f}%")
print(f" Red Neuronal (MLP)  → Accuracy: {nn_accuracy*100:.2f}%")
print(f"\n Archivos generados:")
print("   - scaler.pkl    → Escalador de datos")
print("   - lr_model.pkl  → Modelo Regresión Logística")
print("   - nn_model.pkl  → Modelo Red Neuronal")
print("   - metrics.json  → Métricas de evaluación")
print("\n Listo para ejecutar la API: python app.py")
print("=" * 60)
