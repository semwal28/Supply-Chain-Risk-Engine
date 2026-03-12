import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

def prepare_pytorch_data(df, window_size=7, batch_size=16):
    # 1. Resample to Daily Sales
    df['order_date'] = pd.to_datetime(df['order_date'])
    daily_series = df.groupby('order_date')['Sales'].sum().resample('D').sum().fillna(0)
    values = daily_series.values.reshape(-1, 1)
    
    # 2. Scale
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(values)
    
    # 3. Create Sliding Windows
    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i, 0])
        y.append(scaled_data[i, 0])
    
    # 4. Convert to PyTorch Tensors
    X_tensor = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1) # [Batch, Seq, Feature]
    y_tensor = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1)
    
    # 5. Create DataLoader (Handles Batching)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return loader, scaler