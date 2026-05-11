"""
=============================================================================
PROYECTO FINAL: CLASIFICACIÓN DE FLORES IRIS
=============================================================================
Archivo: app.py
Propósito: API REST con Flask para servir predicciones de los modelos ML

¿QUÉ ES UNA API REST?
----------------------
API = Application Programming Interface (Interfaz de Programación de Aplicaciones)
REST = Representational State Transfer

Una API REST permite que diferentes aplicaciones se comuniquen entre sí
a través de HTTP, usando JSON como formato de datos.

El frontend (HTML/JS) enviará peticiones HTTP a esta API y recibirá
predicciones como respuesta JSON.

FLUJO:
  Frontend (index.html) → HTTP Request → app.py → Modelo ML → HTTP Response → Frontend
=============================================================================
"""

# --- IMPORTACIONES ---
from flask import Flask, request, jsonify  # Framework web ligero para Python
from flask_cors import CORS               # Permite peticiones desde el frontend
import joblib                             # Cargar modelos guardados con joblib
import numpy as np                        # Operaciones numéricas
import json                               # Leer el archivo de métricas

# =============================================================================
# INICIALIZACIÓN DE LA APLICACIÓN FLASK
# =============================================================================

# Flask(__name__) crea la aplicación. __name__ le dice a Flask dónde buscar recursos
app = Flask(__name__)

# CORS (Cross-Origin Resource Sharing) → permite que el HTML pueda llamar a esta API
# Sin CORS, el navegador bloquea peticiones de un dominio a otro por seguridad
# allow_headers=['Content-Type'] → permitimos cabeceras JSON
CORS(app, resources={r"/*": {"origins": "*"}})

# =============================================================================
# CARGAR MODELOS Y SCALER AL INICIAR EL SERVIDOR
# =============================================================================
# Los cargamos UNA SOLA VEZ al iniciar → mucho más eficiente
# que cargarlos en cada petición

try:
    # Cargar el escalador (debe ser el MISMO que se usó en el entrenamiento)
    scaler = joblib.load('scaler.pkl')
    
    # Cargar modelo de Regresión Logística
    lr_model = joblib.load('lr_model.pkl')
    
    # Cargar Red Neuronal
    nn_model = joblib.load('nn_model.pkl')
    
    # Cargar métricas pre-calculadas del entrenamiento
    with open('metrics.json', 'r') as f:
        metrics_data = json.load(f)
    
    # Nombres de las clases para la respuesta
    CLASS_NAMES = metrics_data['class_names']  # ['setosa', 'versicolor', 'virginica']
    
    print("Modelos cargados correctamente")
    print(f"   Clases: {CLASS_NAMES}")
    
except FileNotFoundError as e:
    print(f"❌ ERROR: {e}")
    print("   Asegúrate de ejecutar 'python train_models.py' primero")
    raise  # Re-lanza el error para detener el servidor

# =============================================================================
# ENDPOINT 1: PREDICCIÓN INDIVIDUAL
# Ruta: POST /predict
# =============================================================================
@app.route('/predict', methods=['POST'])
def predict():
    """
    PREDICCIÓN INDIVIDUAL
    ---------------------
    Recibe las 4 mediciones de UNA flor y devuelve la predicción de ambos modelos.
    
    REQUEST (lo que envía el frontend):
        Content-Type: application/json
        Body: {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
    
    RESPONSE (lo que devuelve la API):
        {
            "logistic_regression": {"class": "setosa", "class_id": 0, "probabilities": {...}},
            "neural_network": {"class": "setosa", "class_id": 0, "probabilities": {...}},
            "input_features": {...}
        }
    
    @app.route → decorador que asocia la URL /predict con esta función
    methods=['POST'] → solo acepta peticiones POST (enviamos datos al servidor)
    """
    try:
        # Obtener los datos JSON del cuerpo de la petición
        # request.get_json() deserializa el JSON a diccionario Python
        data = request.get_json()
        
        # Validar que llegaron todos los campos necesarios
        required_fields = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        for field in required_fields:
            if field not in data:
                # Retornamos error 400 (Bad Request) si falta algún campo
                return jsonify({"error": f"Campo faltante: {field}"}), 400
        
        # Extraer los valores y crear el array de features
        # El orden IMPORTA: debe ser el mismo que en el entrenamiento
        features = np.array([[
            float(data['sepal_length']),  # Convertimos a float por seguridad
            float(data['sepal_width']),
            float(data['petal_length']),
            float(data['petal_width'])
        ]])
        # features.shape → (1, 4): 1 muestra, 4 características
        
        # ESCALAR los datos con el MISMO scaler del entrenamiento
        # Si no escalamos, las predicciones serán incorrectas
        features_scaled = scaler.transform(features)
        
        # --- PREDICCIÓN CON REGRESIÓN LOGÍSTICA ---
        lr_pred = lr_model.predict(features_scaled)[0]  # [0] → primer (único) resultado
        # predict_proba devuelve las probabilidades de cada clase [P(setosa), P(versicolor), P(virginica)]
        lr_proba = lr_model.predict_proba(features_scaled)[0]
        
        # --- PREDICCIÓN CON RED NEURONAL ---
        nn_pred = nn_model.predict(features_scaled)[0]
        nn_proba = nn_model.predict_proba(features_scaled)[0]
        
        # Construir la respuesta JSON
        response = {
            "logistic_regression": {
                "class": CLASS_NAMES[lr_pred],           # Nombre de la clase ("setosa")
                "class_id": int(lr_pred),                 # ID numérico (0, 1 o 2)
                "probabilities": {                        # Probabilidades por clase
                    CLASS_NAMES[i]: round(float(lr_proba[i]) * 100, 2)
                    for i in range(len(CLASS_NAMES))      # {"setosa": 98.5, ...}
                }
            },
            "neural_network": {
                "class": CLASS_NAMES[nn_pred],
                "class_id": int(nn_pred),
                "probabilities": {
                    CLASS_NAMES[i]: round(float(nn_proba[i]) * 100, 2)
                    for i in range(len(CLASS_NAMES))
                }
            },
            "input_features": {  # Devolvemos los inputs para confirmar al usuario
                "sepal_length": float(data['sepal_length']),
                "sepal_width": float(data['sepal_width']),
                "petal_length": float(data['petal_length']),
                "petal_width": float(data['petal_width'])
            }
        }
        
        # jsonify convierte el diccionario Python a JSON y retorna código HTTP 200 (OK)
        return jsonify(response), 200
    
    except ValueError as e:
        # Error si los valores no son números válidos
        return jsonify({"error": f"Valores inválidos: {str(e)}"}), 400
    except Exception as e:
        # Error genérico del servidor
        return jsonify({"error": f"Error interno: {str(e)}"}), 500


# =============================================================================
# ENDPOINT 2: PREDICCIÓN POR LOTES (BATCH)
# Ruta: POST /predict_batch
# =============================================================================
@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    PREDICCIÓN POR LOTES
    --------------------
    Recibe MÚLTIPLES filas de datos y devuelve predicciones para todas.
    Útil para clasificar muchas flores a la vez sin hacer múltiples peticiones.
    
    REQUEST:
        Body: {
            "data": [
                [5.1, 3.5, 1.4, 0.2],   ← primera flor: [sl, sw, pl, pw]
                [6.7, 3.0, 5.2, 2.3],   ← segunda flor
                [5.8, 2.7, 4.1, 1.0]    ← tercera flor
            ]
        }
    
    RESPONSE:
        {
            "predictions": [
                {"index": 0, "logistic_regression": {...}, "neural_network": {...}},
                ...
            ],
            "total": 3
        }
    """
    try:
        data = request.get_json()
        
        # Validar que llega la clave 'data' con una lista
        if 'data' not in data or not isinstance(data['data'], list):
            return jsonify({"error": "Formato inválido. Se espera {'data': [[sl,sw,pl,pw], ...]}"}), 400
        
        if len(data['data']) == 0:
            return jsonify({"error": "La lista está vacía"}), 400
        
        # Convertir la lista de listas a matriz numpy
        # Cada fila es una flor con sus 4 mediciones
        features_batch = np.array(data['data'], dtype=float)
        
        # Validar que cada fila tiene exactamente 4 columnas
        if features_batch.ndim != 2 or features_batch.shape[1] != 4:
            return jsonify({"error": "Cada muestra debe tener exactamente 4 valores"}), 400
        
        # Escalar TODO el batch de una vez (más eficiente que escalar uno por uno)
        features_scaled = scaler.transform(features_batch)
        
        # Predicciones en lote
        lr_preds = lr_model.predict(features_scaled)          # Array de N predicciones
        lr_probas = lr_model.predict_proba(features_scaled)   # Matriz Nx3 de probabilidades
        
        nn_preds = nn_model.predict(features_scaled)
        nn_probas = nn_model.predict_proba(features_scaled)
        
        # Construir lista de resultados
        predictions = []
        for i in range(len(features_batch)):
            predictions.append({
                "index": i,          # Número de muestra (para identificarla)
                "input": {
                    "sepal_length": features_batch[i][0],
                    "sepal_width":  features_batch[i][1],
                    "petal_length": features_batch[i][2],
                    "petal_width":  features_batch[i][3]
                },
                "logistic_regression": {
                    "class": CLASS_NAMES[lr_preds[i]],
                    "class_id": int(lr_preds[i]),
                    "probabilities": {
                        CLASS_NAMES[j]: round(float(lr_probas[i][j]) * 100, 2)
                        for j in range(len(CLASS_NAMES))
                    }
                },
                "neural_network": {
                    "class": CLASS_NAMES[nn_preds[i]],
                    "class_id": int(nn_preds[i]),
                    "probabilities": {
                        CLASS_NAMES[j]: round(float(nn_probas[i][j]) * 100, 2)
                        for j in range(len(CLASS_NAMES))
                    }
                }
            })
        
        return jsonify({
            "predictions": predictions,
            "total": len(predictions),  # Cuántas flores se procesaron
            "class_names": CLASS_NAMES
        }), 200
    
    except ValueError as e:
        return jsonify({"error": f"Error en los datos: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Error interno: {str(e)}"}), 500


# =============================================================================
# ENDPOINT 3: MÉTRICAS DE LOS MODELOS
# Ruta: GET /metrics
# =============================================================================
@app.route('/metrics', methods=['GET'])
def get_metrics():
    """
    MÉTRICAS DE EVALUACIÓN
    ----------------------
    Devuelve las métricas calculadas durante el entrenamiento.
    
    Este endpoint usa GET (no POST) porque solo pedimos datos, no enviamos nada.
    
    REQUEST: GET /metrics (sin body)
    
    RESPONSE:
        {
            "logistic_regression": {
                "accuracy": 0.9667,
                "accuracy_pct": 96.67,
                "confusion_matrix": [[10,0,0],[0,9,1],[0,0,10]]
            },
            "neural_network": {...},
            "class_names": [...],
            ...
        }
    """
    try:
        # metrics_data ya está cargado en memoria desde el inicio
        return jsonify(metrics_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =============================================================================
# ENDPOINT EXTRA: HEALTH CHECK
# Ruta: GET /health
# =============================================================================
@app.route('/health', methods=['GET'])
def health_check():
    """
    Endpoint de verificación: confirma que el servidor está funcionando.
    Muy útil para debugging y monitoreo.
    """
    return jsonify({
        "status": "running",
        "models_loaded": True,
        "class_names": CLASS_NAMES,
        "endpoints": ["/predict", "/predict_batch", "/metrics", "/health"]
    }), 200


# =============================================================================
# PUNTO DE ENTRADA: INICIAR EL SERVIDOR
# =============================================================================
if __name__ == '__main__':
    """
    if __name__ == '__main__': → este bloque solo se ejecuta cuando corres
    el archivo directamente con 'python app.py'
    No se ejecuta si el archivo es importado como módulo.
    
    app.run() inicia el servidor Flask:
      - host='0.0.0.0' → acepta conexiones de cualquier IP (no solo localhost)
      - port=5000 → puerto donde escucha el servidor
      - debug=True → reinicia automáticamente al hacer cambios (solo para desarrollo)
    """
    print("\n" + "=" * 60)
    print("Iniciando servidor Flask...")
    print("=" * 60)
    print("📡 Endpoints disponibles:")
    print("   POST http://localhost:5000/predict")
    print("   POST http://localhost:5000/predict_batch")
    print("   GET  http://localhost:5000/metrics")
    print("   GET  http://localhost:5000/health")
    print("\nServidor listo. Abre index.html en tu navegador.")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
