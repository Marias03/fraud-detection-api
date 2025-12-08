# REEMPLAZA tu Streamlit actual con esta versión mejorada:
import streamlit as st
import pandas as pd
import requests
import json

st.title("🏦 Fraud Detection System")
st.markdown("Real-time fraud detection using Machine Learning API")

# Sidebar para configuración
with st.sidebar:
    st.header("🔧 Configuration")
    api_url = st.text_input("API URL", "http://localhost:8000")
    st.markdown("---")
    
    # Botón para ver estado del API
    if st.button("Check API Status"):
        try:
            response = requests.get(f"{api_url}/health")
            if response.status_code == 200:
                st.success("✅ API is running")
            else:
                st.error("❌ API not responding")
        except:
            st.error("❌ Cannot connect to API")

st.subheader("📊 Enter Transaction Details")

# Formulario en dos columnas
col1, col2 = st.columns(2)

with col1:
    transaction_type = st.selectbox(
        "Transaction Type",
        ["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"]
    )
    
    amount = st.number_input(
        "Amount ($)",
        min_value=0.0,
        value=5000.0,
        step=100.0
    )
    
    oldbalanceOrg = st.number_input(
        "Old Balance (Sender)",
        min_value=0.0,
        value=10000.0,
        step=100.0
    )
    
    newbalanceOrig = st.number_input(
        "New Balance (Sender)",
        min_value=0.0,
        value=5000.0,
        step=100.0
    )
    
    nameDest = st.text_input(
        "Destination Account ID",
        value="C1234567890"
    )

with col2:
    oldbalanceDest = st.number_input(
        "Old Balance (Receiver)",
        min_value=0.0,
        value=0.0,
        step=100.0
    )
    
    newbalanceDest = st.number_input(
        "New Balance (Receiver)",
        min_value=0.0,
        value=5000.0,
        step=100.0
    )
    
    # Calcular diferencias automáticamente
    balanceDiffOring = oldbalanceOrg - newbalanceOrig
    balanceDiffDest = newbalanceDest - oldbalanceDest
    balanceDiffOrig = oldbalanceOrg - newbalanceOrig  # Puede ser diferente
    
    st.metric("Balance Diff (Sender)", f"${balanceDiffOring:,.2f}")
    st.metric("Balance Diff (Receiver)", f"${balanceDiffDest:,.2f}")

# Botón de predicción
if st.button("🔍 Analyze Transaction", type="primary"):
    # Preparar datos para el API
    transaction_data = {
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "nameDest": nameDest,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "balanceDiffOring": balanceDiffOring,
        "balanceDiffDest": balanceDiffDest,
        "balanceDiffOrig": balanceDiffOrig
    }
    
    with st.spinner("Analyzing transaction..."):
        try:
            # Llamar al API
            response = requests.post(
                f"{api_url}/predict",
                json=transaction_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Mostrar resultados
                st.divider()
                st.subheader("📋 Analysis Results")
                
                # Tarjeta de resultado
                if result["is_fraud"]:
                    st.error(f"🚨 {result['message']}")
                    st.metric("Fraud Probability", f"{result.get('fraud_probability', 0)*100:.1f}%")
                else:
                    st.success(f"✅ {result['message']}")
                    st.metric("Fraud Probability", f"{result.get('fraud_probability', 0)*100:.1f}%")
                
                # Detalles técnicos (expandible)
                with st.expander("Technical Details"):
                    st.json(result)
                    
                # Mostrar datos enviados
                with st.expander("Transaction Data Sent"):
                    st.json(transaction_data)
                    
            else:
                st.error(f"API Error: {response.status_code}")
                st.code(response.text)
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Make sure it's running.")
            st.info("Run this command in another terminal: `python api.py`")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Sección de información
st.divider()
st.subheader("ℹ️ About This System")

st.markdown("""
**Architecture:**
- **Frontend**: Streamlit application for user interaction
- **Backend**: FastAPI REST API for model serving  
- **ML Model**: Scikit-learn Pipeline for fraud detection

**Key Features:**
- Real-time fraud prediction
- REST API for integration with other systems
- Interactive documentation at `/docs`
- Batch prediction support
- Health monitoring endpoint
""")

# Botón para abrir documentación del API
if st.button("📚 Open API Documentation"):
    st.markdown(f"[API Docs]({api_url}/docs)")