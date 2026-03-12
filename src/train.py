import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
# Importing our previous work
from data_ingestion import load_and_clean_data
from transformation import get_preprocessing_pipeline
from sklearn.pipeline import Pipeline

def run_training():
    print("--- Starting Phase 2: Training Pipeline ---")
    
    # 1. Ingest Data
    df = load_and_clean_data('C:\\Users\\semwa\\OneDrive\\Desktop\\Supply-Chain-Risk-Engine\\data\\DataCoSupplyChainDataset.csv')
    
    # 2. Define X and y
    # We pass the whole dataframe; the Preprocessor will select the columns
    X = df 
    y = df['Late_delivery_risk']

    # 3. Train/Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Build the Full Pipeline
    # This combines our 'Feature Engineering' and 'ML Model' into one object
    model_pipeline = Pipeline(steps=[
        ('preprocessor', get_preprocessing_pipeline()),
        ('classifier', XGBClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=6, 
            random_state=42
        ))
    ])

    # 5. Execute Training
    print("Fitting model to supply chain data...")
    model_pipeline.fit(X_train, y_train)

    # 6. Evaluate Results
    y_pred = model_pipeline.predict(X_test)
    print("\nModel Performance Report:")
    print(classification_report(y_test, y_pred))

    # 7. Save the artifact (The .pkl file is your real Resume asset)
    joblib.dump(model_pipeline, 'C:\\Users\\semwa\\OneDrive\\Desktop\\Supply-Chain-Risk-Engine\\models\\risk_model_v1.pkl')
    print("\nSuccess: Model saved to models/risk_model_v1.pkl")

if __name__ == "__main__":
    run_training()