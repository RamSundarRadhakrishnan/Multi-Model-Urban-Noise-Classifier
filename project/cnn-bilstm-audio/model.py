import torch
import torch.nn as nn


class CNNBiLSTMAudioClassifier(nn.Module):
    def __init__(
        self,
        num_classes,
        n_mfcc=40,
        cnn_channels=64,
        lstm_hidden=128,
        dense_units=64,
        dropout=0.3
    ):
        super().__init__()

        self.n_mfcc = n_mfcc

        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=n_mfcc,
                out_channels=cnn_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout)
        )

        self.bilstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units, num_classes)
        )

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError("Expected input shape [batch, time_steps, n_mfcc]")

        if x.size(2) != self.n_mfcc:
            raise ValueError(f"Expected {self.n_mfcc} MFCC features, got {x.size(2)}")

        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.bilstm(x)

        x = lstm_out.mean(dim=1)
        logits = self.classifier(x)

        return logits