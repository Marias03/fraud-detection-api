# api.py - VERSIÓN FINAL PARA TU MODELO
from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
from pydantic import BaseModel
from typing import Optional, List
import json

print("Loading Fraud detection model...")

try:
    # USAR JOBLIB (como funcionó en tu notebook)
    modelo = joblib.load('fraud_detection_pipeline.pkl')
    
    print("Model loaded successfully!")
    print(f"Type: {type(modelo)}")
    print(f"Expected features ({len(modelo.feature_names_in_)}):")
    for i, feature in enumerate(modelo.feature_names_in_, 1):
        print(f"   {i:2d}. {feature}")
    
except Exception as e:
    print(f"❌ Error cargando el modelo: {e}")
    print("💡 Asegúrate de que 'fraud_detection_pipeline.pkl' esté en la misma carpeta")
    modelo = None


class Transaccion(BaseModel):
    # Las 10 características que tu modelo espera, EN ORDEN:
    type: str  # Tipo de transacción (ej: "CASH_OUT", "PAYMENT", etc.)
    amount: float  # Monto de la transacción
    oldbalanceOrg: float  # Balance inicial origen
    newbalanceOrig: float  # Balance nuevo origen
    nameDest: str  # Nombre del destinatario (puede ser string)
    oldbalanceDest: float  # Balance inicial destino
    newbalanceDest: float  # Balance nuevo destino
    balanceDiffOring: float  # Diferencia de balance origen
    balanceDiffDest: float  # Diferencia de balance destino
    balanceDiffOrig: float  # Otra diferencia de balance origen


app = FastAPI(
    title="API de Detección de Fraude Bancario",
    description="API para predecir si una transacción es fraudulenta usando Machine Learning",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


@app.get("/")
def home():
    """Página de inicio del API"""
    return {
        "message": "Welcome to Fraud detection prediction App",
        "status": "operational" if modelo is not None else "model_not_loaded",
        "endpoints": {
            "GET /": "this page",
            "GET /info": "Model information",
            "GET /features": "Características requeridas",
            "POST /predict": "Hacer una predicción",
            "POST /predict_batch": "Múltiples predicciones",
            "GET /health": "Estado del servicio"
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        }
    }

@app.get("/health")
def health_check():
    """Verificar estado del API"""
    return {
        "status": "healthy",
        "model_loaded": modelo is not None,
        "timestamp": pd.Timestamp.now().isoformat()
    }

@app.get("/info")
def model_info():
    """Información detallada del modelo"""
    if modelo is None:
        return {"error": "Modelo no cargado"}
    
    info = {
        "model_type": str(type(modelo)),
        "n_features": len(modelo.feature_names_in_),
        "features": modelo.feature_names_in_.tolist(),
        "pipeline_steps": []
    }
    
    # Obtener información de los pasos del pipeline
    if hasattr(modelo, 'steps'):
        for step_name, step_obj in modelo.steps:
            step_info = {
                "name": step_name,
                "type": str(type(step_obj)),
                "parameters": list(step_obj.get_params().keys())[:5]  # Primeros 5 params
            }
            info["pipeline_steps"].append(step_info)
    
    return info

@app.get("/features")
def get_features():
    """Mostrar las características que necesita el modelo"""
    if modelo is None:
        return {"error": "Modelo no cargado"}
    
    features = modelo.feature_names_in_.tolist()
    
    # Crear ejemplo con valores típicos
    ejemplo = {
        "type": "CASH_OUT",  # Valores comunes: CASH_OUT, PAYMENT, CASH_IN, TRANSFER, DEBIT
        "amount": 5000.0,
        "oldbalanceOrg": 10000.0,
        "newbalanceOrig": 5000.0,
        "nameDest": "C1234567890",  # Parece un ID de cliente
        "oldbalanceDest": 0.0,
        "newbalanceDest": 5000.0,
        "balanceDiffOring": 5000.0,  # oldbalanceOrg - newbalanceOrig
        "balanceDiffDest": 5000.0,   # newbalanceDest - oldbalanceDest
        "balanceDiffOrig": 5000.0    # Posiblemente otra métrica
    }
    
    return {
        "required_features": features,
        "example": ejemplo,
        "note": "Los nombres deben coincidir exactamente (case-sensitive)"
    }

@app.post("/predict")
def predict_single(transaccion: Transaccion):
    """
    Predecir si UNA transacción es fraudulenta
    
    Ejemplo de cuerpo de solicitud:
    ```json
    {
        "type": "CASH_OUT",
        "amount": 5000.0,
        "oldbalanceOrg": 10000.0,
        "newbalanceOrig": 5000.0,
        "nameDest": "C1234567890",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 5000.0,
        "balanceDiffOring": 5000.0,
        "balanceDiffDest": 5000.0,
        "balanceDiffOrig": 5000.0
    }
    ```
    """
    if modelo is None:
        return {"error": "model not loaded"}
    
    try:
        # Convertir a diccionario
        datos = transaccion.dict()
        
        # Crear DataFrame (IMPORTANTE: en el orden correcto)
        df = pd.DataFrame([datos])
        
        # Asegurar el orden de columnas
        df = df[modelo.feature_names_in_]
        
        # Hacer predicción
        prediccion = modelo.predict(df)[0]
        
        # Obtener probabilidades (si el modelo lo soporta)
        probabilidad_fraude = None
        if hasattr(modelo, 'predict_proba'):
            proba = modelo.predict_proba(df)[0]
            # Asumimos que clase 1 es fraude (ajusta si es diferente)
            probabilidad_fraude = float(proba[1])
        
        # Interpretar resultado
        es_fraude = bool(prediccion)
        mensaje = "Alert:Potentially fraudulent transaction" if es_fraude else "Legitimate transaction"
        
        # Preparar respuesta
        respuesta = {
            "transaction_id": datos.get("nameDest", "N/A"),
            "is_fraud": es_fraude,
            "prediction": int(prediccion),
            "fraud_probability": probabilidad_fraude,
            "message": mensaje,
            "confidence": "high" if probabilidad_fraude and probabilidad_fraude > 0.8 else "medium",
            "status": "success",
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        # Log (útil para debugging)
        print(f"prediction made: {mensaje}")
        if probabilidad_fraude:
            print(f"   Probabilidad de fraude: {probabilidad_fraude:.2%}")
        
        return respuesta
        
    except Exception as e:
        print(f"❌ Error en predicción: {str(e)}")
        return {
            "error": str(e),
            "status": "error",
            "suggestion": "Verifica que los nombres y tipos de datos coincidan con /features"
        }

@app.post("/predict_batch")
def predict_batch(transacciones: List[Transaccion]):
    """
    Predecir MÚLTIPLES transacciones a la vez
    
    Ejemplo:
    ```json
    [
        {
            "type": "CASH_OUT",
            "amount": 5000.0,
            ...
        },
        {
            "type": "PAYMENT", 
            "amount": 100.0,
            ...
        }
    ]
    ```
    """
    if modelo is None:
        return {"error": "Failed on detection model"}
    
    try:
        # Convertir todas las transacciones a DataFrame
        datos_list = [t.dict() for t in transacciones]
        df = pd.DataFrame(datos_list)
        
        # Asegurar orden de columnas
        df = df[modelo.feature_names_in_]
        
        # Hacer predicciones
        predicciones = modelo.predict(df)
        
        # Probabilidades si están disponibles
        probabilidades = None
        if hasattr(modelo, 'predict_proba'):
            probabilidades = modelo.predict_proba(df)
        
        # Preparar resultados
        resultados = []
        for i, (idx, pred) in enumerate(zip(df.index, predicciones)):
            resultado = {
                "transaction_index": i,
                "transaction_id": datos_list[i].get("nameDest", f"tx_{i}"),
                "is_fraud": bool(pred),
                "prediction": int(pred),
                "fraud_probability": float(probabilidades[i][1]) if probabilidades is not None else None,
                "type": datos_list[i]["type"],
                "amount": datos_list[i]["amount"]
            }
            resultados.append(resultado)
        
        # Estadísticas
        n_fraudes = sum(predicciones)
        stats = {
            "total_transactions": len(predicciones),
            "fraudulent_count": int(n_fraudes),
            "fraudulent_percentage": float(n_fraudes / len(predicciones)) if len(predicciones) > 0 else 0,
            "legitimate_count": int(len(predicciones) - n_fraudes)
        }
        
        return {
            "results": resultados,
            "statistics": stats,
            "status": "success",
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "status": "error",
            "transactions_received": len(transacciones)
        }

@app.post("/predict_simple")
def predict_simple_json(data: dict):
    """
    Versión simple que acepta JSON directamente
    Útil para pruebas rápidas desde Postman o curl
    """
    if modelo is None:
        return {"error": "Failed"}
    
    try:
        df = pd.DataFrame([data])
        df = df[modelo.feature_names_in_]
        
        prediccion = modelo.predict(df)[0]
        es_fraude = bool(prediccion)
        
        return {
            "is_fraud": es_fraude,
            "prediction": int(prediccion),
            "input_received": data
        }
    except Exception as e:
        return {
            "error": str(e),
            "received_data": data
        }

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Obtener puerto de Railway o usar 8000 por defecto
    port = int(os.environ.get("PORT", 8000))
    
    print("\n" + "="*50)
    print("Fraud detection API - Ready for deployment on Railway")
    print("="*50)
    print(f"\n📡 Server starting on port {port}")
    print("Available endpoints:")
    print(f"  • http://0.0.0.0:{port}/          - Página principal")
    print(f"  • http://0.0.0.0:{port}/docs      - Documentación interactiva")
    print(f"  • http://0.0.0.0:{port}/features  - Características requeridas")
    print(f"  • http://0.0.0.0:{port}/health    - Estado del servicio")
    print("\n⚡ Starting server...")
    
    uvicorn.run(
        app="api:app",
        host="0.0.0.0",
        port=port,
        reload=False,     # ✅ IMPORTANTE: False en producción
        log_level="info"
    )