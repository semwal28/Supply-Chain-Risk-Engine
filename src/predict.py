import joblib
import pandas as pd

def predict_risk(input_data):
    """
    Takes a dictionary of shipment details and returns the risk level.
    """
    # 1. Load the trained pipeline (Preprocessor + XGBoost)
    model = joblib.load('C:\\Users\\semwa\\OneDrive\\Desktop\\Supply-Chain-Risk-Engine\\models\\risk_model_v1.pkl')
    
    # 2. Convert input dictionary to DataFrame
    # Note: Column names must be lowercase to match our sanitized training data
    
    df_input = pd.DataFrame([input_data])

    # 3. Make Prediction
    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0][1] # Probability of being 'Late'
    
    return prediction, probability

if __name__ == "__main__":
    # --- SIMULATE A NEW SHIPMENT ---
    # This data mimics a new order coming into Dell's system
    new_order = {
        'Type': 'TRANSFER',
        'Days_for_shipment': 2,
        'Category_Name': 'Sporting Goods',
        'Order_Region': 'Southeast Asia',
        'Shipping_Mode': 'Standard Class',
        'Customer_Segment': 'Consumer',
        'Sales': 314.64,
        'Order_Item_Quantity': 1
    }
    
    result, prob = predict_risk(new_order)
            
    print("---Supply Chain Risk Analysis ---")
    print(f"Shipment Data: {new_order['Shipping_Mode']} to {new_order['Order_Region']}")
    print(f"Late Delivery Prediction: {'HIGH RISK (Late)' if result == 1 else 'LOW RISK (On-Time)'}")
    print(f"Risk Probability: {prob:.2%}")