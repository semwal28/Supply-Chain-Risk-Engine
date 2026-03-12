import torch
import torch.nn as nn

class DemandGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(DemandGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # The GRU Layer
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        
        # The Output Layer (Dense)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initializing hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward propagate GRU
        out, _ = self.gru(x, h0)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out