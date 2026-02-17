"""
LSTM Model Architecture for Stampede Risk Prediction
"""

import torch
import torch.nn as nn


class StampedeRiskLSTM(nn.Module):
    """LSTM model for stampede risk prediction"""
    
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, output_size=1):
        """
        Initialize LSTM model
        
        Args:
            input_size: Number of input features (default: 6)
                - person_count, density, avg_bbox_area, flow_mean, flow_std, flow_max
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            output_size: Output size (1 for risk score)
        """
        super(StampedeRiskLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Fully connected layers using Sequential to match saved model structure
        # fc.0 = Linear, fc.1 = ReLU, fc.2 = Dropout, fc.3 = Linear, fc.4 = Sigmoid
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),      # fc.0
            nn.ReLU(),                        # fc.1
            nn.Dropout(0.2),                  # fc.2
            nn.Linear(32, output_size),       # fc.3
            nn.Sigmoid()                      # fc.4
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Risk score between 0 and 1
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.fc(last_output)
        
        return out
