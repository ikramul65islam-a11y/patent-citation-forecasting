"""Citation-history-only GRU architecture.

Inputs are the five log-transformed annual citation counts and four citation
summary features. Data splitting, scaling, selected settings, and training are
handled by the experiment runner in ``src/models/full_panel_xgboost_gru.py``.
"""
from torch import nn
import torch


class CitationOnlyGRU(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = int(config["gru_hidden"])
        layers = int(config["gru_layers"])
        head = int(config["head_hidden"])
        dropout = float(config["dropout"])
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        second = max(32, head // 2)
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden + 4, head), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head, second), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(second, 1),
        )

    def forward(self, sequence, scalars):
        if sequence.ndim != 2 or sequence.shape[1] != 5:
            raise ValueError("sequence must have shape (batch, 5)")
        if scalars.ndim != 2 or scalars.shape[1] != 4:
            raise ValueError("scalars must have shape (batch, 4)")
        _, state = self.gru(sequence.unsqueeze(-1))
        return self.prediction_head(torch.cat((state[-1], scalars), dim=1)).squeeze(-1)
