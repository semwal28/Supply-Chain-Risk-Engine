import torch
import joblib
import pandas as pd
import numpy as np
from src.dl_model import DemandGRU

def run_stress_test():
    # Load Models
    risk_model = joblib.load('models/risk_model_v1.pkl')
    demand_model = DemandGRU()
    demand_model.load_state_dict(torch.load('models/demand_gru_v1.pth'))
    demand_model.eval()
    scaler = joblib.load('models/timeseries_scaler.pkl')

    print("="*50)
    print("SCENARIO 1: SHIPPING MODE SENSITIVITY")
    print("="*50)
    
    # Base data (everything is the same except shipping mode)
    test_order_fast = {
        'Type': 'TRANSFER',
        'Days_for_shipment': 1, # Very tight deadline
        'Category_Name': 'Sporting Goods',
        'Order_Region': 'Southeast Asia',
        'Shipping_Mode': 'Same Day',      # <--- Change this to 'Same Day'
        'Customer_Segment': 'Consumer',
        'Sales': 314.64,
        'Order_Item_Quantity': 1
      }

    for mode in ['Standard_Class', 'Second_Class', 'First_Class', 'Same_Day']:
        test_order = test_order_fast.copy()
        test_order['Shipping_Mode'] = mode
        prob = risk_model.predict_proba(pd.DataFrame([test_order]))[0][1]
        print(f"Mode: {mode:15} | Risk Probability: {prob:.2%}")

    print("\n" + "="*50)
    print("SCENARIO 2: DEMAND GROWTH PROJECTION")
    print("="*50)

    # Test Case A: Stable Sales ($5000 every day)
    stable_history = [5000] * 7
    # Test Case B: Rapid Growth (Sales doubling)
    growth_history = [1000, 2000, 3000, 4000, 5000, 6000, 7000]

    for name, history in [("Stable", stable_history), ("Growth", growth_history)]:
        scaled = scaler.transform(np.array(history).reshape(-1, 1))
        input_t = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            pred = demand_model(input_t)
            forecast = scaler.inverse_transform(pred.numpy())[0][0]
        
        print(f"Trend: {name:7} | Last Day: ${history[-1]} | Predicted Next: ${forecast:.2f}")

if __name__ == "__main__":
    run_stress_test()