"""Citation plus scientific-paper activity GRU.

This model definition is included from the user's paper-GRU notebook source,
as requested. It takes a five-step, three-channel sequence: log citation count,
log matched-paper count, and log keyword-hit count. Five scalar inputs contain
the citation summaries plus the late-source-coverage indicator. The source
notebook supplies the training procedure and data preparation; this file does
not claim to be a standalone reproduction of the final experiment.
"""
from torch import nn
import torch


class PaperCitationGRU(nn.Module):
    def __init__(self, config, sequence_features=3, scalar_features=5):
        super().__init__()
        hidden = int(config["gru_hidden"])
        layers = int(config["gru_layers"])
        head = int(config["head_hidden"])
        dropout = float(config["dropout"])
        self.gru = nn.GRU(
            input_size=sequence_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        second = max(32, head // 2)
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden + scalar_features, head), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head, second), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(second, 1),
        )

    def forward(self, sequence, scalars):
        if sequence.ndim != 3 or sequence.shape[1:] != (5, 3):
            raise ValueError("sequence must have shape (batch, 5, 3)")
        if scalars.ndim != 2 or scalars.shape[1] != 5:
            raise ValueError("scalars must have shape (batch, 5)")
        _, state = self.gru(sequence)
        return self.prediction_head(torch.cat((state[-1], scalars), dim=1)).squeeze(-1)
