import torch
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.model_selection import train_test_split

# Import your custom modules
from dl_model import DemandGRU
from data_ingestion import load_and_clean_data

def test_ml_risk_model():
    print("\n" + "="*20)
    print("TESTING ML RISK MODEL (XGBoost)")
    print("="*20)
    
    # 1. Load data and model
    df = load_and_clean_data('C:\\Users\\semwa\\OneDrive\\Desktop\\Supply-Chain-Risk-Engine\\data\\DataCoSupplyChainDataset.csv')
    model = joblib.load('C:\\Users\\semwa\\OneDrive\\Desktop\\Supply-Chain-Risk-Engine\\models\\risk_model_v1.pkl')
    
    # 2. Recreate the EXACT same test split used in Phase 2
    X = df
    y = df['Late_delivery_risk']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Predict and Evaluate
    y_pred = model.predict(X_test)
    print("Model loaded successfully. Performance on Unseen Data:")
    print(classification_report(y_test, y_pred))

def test_dl_demand_model():
    print("\n" + "="*40)
    print("TESTING DL DEMAND MODEL (PyTorch)")
    print("="*40)
    
    # 1. Load model architecture, weights, and scaler
    model = DemandGRU()
    model.load_state_dict(torch.load('C:\\Users\\semwa\\OneDrive\\Desktop\\Supply-Chain-Risk-Engine\\models\\demand_gru_v1.pth'))
    model.eval() # Set to evaluation mode (turns off Dropout)
    scaler = joblib.load('C:\\Users\\semwa\\OneDrive\\Desktop\\Supply-Chain-Risk-Engine\\models\\timeseries_scaler.pkl')
    
    # 2. Prepare a "Real-World" sequence (the last 7 days of sales)
    df = load_and_clean_data('C:\\Users\\semwa\\OneDrive\\Desktop\\Supply-Chain-Risk-Engine\\data\\DataCoSupplyChainDataset.csv')
    df['order_date'] = pd.to_datetime(df['order_date'])
    daily_sales = df.groupby('order_date')['Sales'].sum().resample('D').sum().fillna(0)
    
    # Take the last 7 days from the dataset to predict the 8th day
    last_7_days = daily_sales.values[-7:].reshape(-1, 1)
    scaled_input = scaler.transform(last_7_days)
    input_tensor = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0) # Add batch dimension
    
    # 3. Forecast
    with torch.no_grad():
        prediction_scaled = model(input_tensor)
        prediction_actual = scaler.inverse_transform(prediction_scaled.numpy())
    
    print(f"Input Sales (Last 3 days): {daily_sales.values[-3:].tolist()}")
    print(f"Predicted Sales for Next Day: ${prediction_actual[0][0]:.2f}")
    print("PyTorch Inference: SUCCESS")

if __name__ == "__main__":
    try:
        test_ml_risk_model()
        test_dl_demand_model()
    except Exception as e:
        print(f"TEST FAILED: {e}")