import pandas as pd
import numpy as np
import os

def load_and_clean_data(file_path):
    # Load with specific encoding due to special characters in city names
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    df.columns = df.columns.str.strip()
    
    # Drop columns that are useless for ML (like URLs and Passwords)
    cols_to_drop = ['Customer_Email', 'Customer_Password', 'Product_Image', 'Customer_Fname', 'Customer_Lname']
    df = df.drop(columns=cols_to_drop)
    
    # Handle dates: Convert strings to datetime objects
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['shipping_date'] = pd.to_datetime(df['shipping_date'])
    
    # Fill missing values for ZIP codes (common in this dataset)
    df['Order_Zipcode'] = df['Order_Zipcode'].fillna(0)
    
    print(f"Data Ingested: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.columns.tolist())
    return df

if __name__ == "__main__":
    # Test the ingestion
    raw_data_path = r'C:\Users\semwa\OneDrive\Desktop\Supply-Chain-Risk-Engine\data\DataCoSupplyChainDataset.csv'
    clean_df = load_and_clean_data(raw_data_path)
    # Save a small sample to the data folder for inspection
    clean_df.head().to_csv(r'C:\Users\semwa\OneDrive\Desktop\Supply-Chain-Risk-Engine\data\cleaned_sample.csv', index=False)