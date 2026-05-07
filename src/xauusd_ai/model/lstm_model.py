"""
LSTM Model for Temporal Pattern Recognition

Part of P1 Ensemble technique - combines with LightGBM for improved predictions.
LSTM captures sequential dependencies that tree-based models miss.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available - LSTM model disabled")


if TORCH_AVAILABLE:
    class LSTMClassifier(nn.Module):
        """Simple LSTM binary classifier for XAUUSD signals."""
        
        def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            
            self.fc1 = nn.Linear(hidden_size, 32)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.fc2 = nn.Linear(32, 2)  # Binary classification
            
        def forward(self, x):
            # x shape: (batch, seq_len, features)
            lstm_out, _ = self.lstm(x)
            # Take last timestep output
            last_out = lstm_out[:, -1, :]
            out = self.fc1(last_out)
            out = self.relu(out)
            out = self.dropout(out)
            out = self.fc2(out)
            return out


    class LSTMModelWrapper:
        """Scikit-learn compatible wrapper for LSTM model."""
        
        def __init__(
            self,
            sequence_length: int = 20,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.2,
            learning_rate: float = 0.001,
            epochs: int = 50,
            batch_size: int = 64,
            device: str | None = None,
        ):
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch required for LSTM model")
                
            self.sequence_length = sequence_length
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.dropout = dropout
            self.learning_rate = learning_rate
            self.epochs = epochs
            self.batch_size = batch_size
            
            if device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)
                
            self.model: LSTMClassifier | None = None
            self.input_size: int | None = None
            self.classes_ = np.array([0, 1])  # For sklearn compatibility
            
        def _create_sequences(self, X: np.ndarray, y: np.ndarray | None = None):
            """Convert tabular data to sequences for LSTM."""
            sequences = []
            labels = [] if y is not None else None
            
            for i in range(len(X) - self.sequence_length + 1):
                seq = X[i:i + self.sequence_length]
                sequences.append(seq)
                if y is not None:
                    # Use label at the end of sequence
                    labels.append(y[i + self.sequence_length - 1])
                    
            sequences = np.array(sequences, dtype=np.float32)
            if labels is not None:
                labels = np.array(labels, dtype=np.int64)
                return sequences, labels
            return sequences
            
        def fit(self, X: np.ndarray, y: np.ndarray):
            """Train LSTM model."""
            # Create sequences
            X_seq, y_seq = self._create_sequences(X, y)
            
            if len(X_seq) == 0:
                raise ValueError(f"Not enough data for sequence_length={self.sequence_length}")
                
            # Initialize model
            self.input_size = X.shape[1]
            self.model = LSTMClassifier(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
            ).to(self.device)
            
            # Prepare data
            X_tensor = torch.from_numpy(X_seq).to(self.device)
            y_tensor = torch.from_numpy(y_seq).to(self.device)
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            # Training setup
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            # Training loop
            self.model.train()
            for epoch in range(self.epochs):
                total_loss = 0.0
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    
                if (epoch + 1) % 10 == 0:
                    avg_loss = total_loss / len(dataloader)
                    logging.info(f"LSTM Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")
                    
            return self
            
        def predict(self, X: np.ndarray) -> np.ndarray:
            """Predict class labels."""
            proba = self.predict_proba(X)
            return (proba[:, 1] > 0.5).astype(int)
            
        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            """Predict class probabilities."""
            if self.model is None:
                raise ValueError("Model not trained yet")
                
            # Create sequences (without labels)
            X_seq = self._create_sequences(X, y=None)
            
            if len(X_seq) == 0:
                # Not enough data for sequences - return neutral probabilities
                return np.full((len(X), 2), 0.5)
                
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.from_numpy(X_seq).to(self.device)
                outputs = self.model(X_tensor)
                # Apply softmax to get probabilities
                proba = torch.softmax(outputs, dim=1).cpu().numpy()
                
            # Pad predictions for rows that couldn't form sequences
            n_pad = len(X) - len(proba)
            if n_pad > 0:
                # Prepend neutral probabilities for initial rows
                pad_proba = np.full((n_pad, 2), 0.5)
                proba = np.vstack([pad_proba, proba])
                
            return proba
else:
    # Dummy classes when PyTorch not available
    LSTMClassifier = None
    LSTMModelWrapper = None


def create_lstm_model(
    input_dim: int | None = None,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
    epochs: int = 50,
    batch_size: int = 64,
) -> LSTMModelWrapper | None:
    """
    Factory function to create LSTM model.
    
    Args:
        input_dim: Input feature dimension (not used - inferred during fit())
        hidden_dim: LSTM hidden units
        num_layers: LSTM depth
        dropout: Dropout rate
        learning_rate: Adam learning rate
        epochs: Training epochs
        batch_size: Batch size
    
    Returns:
        LSTMModelWrapper instance or None if PyTorch not available
    """
    if not TORCH_AVAILABLE:
        logging.getLogger(__name__).warning("PyTorch not available - skipping LSTM model")
        return None
        
    return LSTMModelWrapper(
        sequence_length=20,  # Use 20 bars lookback for temporal patterns
        hidden_size=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
    )
