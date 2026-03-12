import torch
import torch.nn as nn
import torch.optim as optim
from dl_preprocess import prepare_pytorch_data
from dl_model import DemandGRU
from data_ingestion import load_and_clean_data
import joblib

def train_pytorch_model():
    print("--- Phase 3: PyTorch GRU Training ---")
    df = load_and_clean_data('C:\\Users\\semwa\\OneDrive\\Desktop\\Supply-Chain-Risk-Engine\\data\\DataCoSupplyChainDataset.csv')
    train_loader, scaler = prepare_pytorch_data(df)
    
    model = DemandGRU()
    criterion = nn.MSELoss() # Loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training Loop
    epochs = 20
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()      # Clear gradients
            outputs = model(batch_X)    # Forward pass
            loss = criterion(outputs, batch_y)
            loss.backward()             # Backward pass (Backpropagation)
            optimizer.step()            # Update weights
            epoch_loss += loss.item()
            
        if (epoch+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}')

    # Save Model State and Scaler
    torch.save(model.state_dict(), 'C:\\Users\\semwa\\OneDrive\\Desktop\\Supply-Chain-Risk-Engine\\models\\demand_gru_v1.pth')
    joblib.dump(scaler, 'C:\\Users\\semwa\\OneDrive\\Desktop\\Supply-Chain-Risk-Engine\\models\\timeseries_scaler.pkl')
    print("PyTorch model saved to C:models\\demand_gru_v1.pth")

if __name__ == "__main__":
    train_pytorch_model()