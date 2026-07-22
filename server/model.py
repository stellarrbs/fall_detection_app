import torch
import torch.nn as nn


class FallLSTM(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        input_size = (
            cfg["dataset"]["num_keypoints"]
            * cfg["dataset"]["features"]
        )

        hidden_size = cfg["model"]["hidden_size"]
        num_layers = cfg["model"]["num_layers"]
        dropout = cfg["model"]["dropout"]
        num_classes = len(cfg["labels"])

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        return self.classifier(last_out)
