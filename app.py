from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import joblib
import pandas as pd
import numpy as np
from src.dl_model import DemandGRU

# 1. Define the Data Schema (This fixes the Swagger UI)
class SupplyChainRequest(BaseModel):
    order_data: dict
    history_data: list

app = FastAPI(title="Dell Supply Chain Intelligence API")

# 2. Global Model Loading
try:
    risk_model = joblib.load('models/risk_model_v1.pkl')
    demand_model = DemandGRU()
    demand_model.load_state_dict(torch.load('models/demand_gru_v1.pth'))
    demand_model.eval()
    scaler = joblib.load('models/timeseries_scaler.pkl')
    print("AI Engines Online.")
except Exception as e:
    print(f"Error loading models: {e}")

@app.get("/")
def home():
    return {"status": "Ready", "model_versions": {"xgboost": "1.2", "pytorch_gru": "1.0"}}

@app.post("/predict")
async def predict_all(request: SupplyChainRequest):
    """
    Unified endpoint for Risk Scoring and Demand Forecasting.
    """
    try:
        # Extract data from the validated request
        order_details = request.order_data
        sales_history = request.history_data

        # --- PART A: ML RISK PREDICTION ---
        df_risk = pd.DataFrame([order_details])
        risk_prob = risk_model.predict_proba(df_risk)[0][1]

        # --- PART B: DL DEMAND FORECAST ---
        scaled_history = scaler.transform(np.array(sales_history).reshape(-1, 1))
        input_tensor = torch.tensor(scaled_history, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            pred_scaled = demand_model(input_tensor)
            forecast = scaler.inverse_transform(pred_scaled.numpy())[0][0]

        return {
            "shipment_analysis": {
                "risk_probability": f"{risk_prob:.2%}",
                "level": "High Risk" if risk_prob > 0.8 else "Low Risk",
                "action": "Flag for Manual Review" if risk_prob > 0.8 else "Auto-Approve"
            },
            "forecast_analysis": {
                "next_day_demand": f"${forecast:.2f}",
                "confidence": "High" if len(sales_history) >= 7 else "Low"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))