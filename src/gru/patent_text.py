"""Citation-history plus frozen patent-text embedding GRU architecture."""
from torch import nn
import torch


class CitationTextGRU(nn.Module):
    def __init__(self, config, text_dim=384):
        super().__init__()
        hidden = int(config["gru_hidden"])
        layers = int(config["gru_layers"])
        head = int(config["head_hidden"])
        dropout = float(config["dropout"])
        projection = int(config["semantic_projection"])
        self.gru = nn.GRU(1, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.text_branch = nn.Sequential(
            nn.Linear(text_dim, projection), nn.ReLU(), nn.Dropout(dropout)
        )
        second = max(32, head // 2)
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden + 4 + projection, head), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head, second), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(second, 1),
        )

    def forward(self, sequence, scalars, text_embedding):
        _, state = self.gru(sequence.unsqueeze(-1))
        text = self.text_branch(text_embedding)
        return self.prediction_head(
            torch.cat((state[-1], scalars, text), dim=1)
        ).squeeze(-1)
