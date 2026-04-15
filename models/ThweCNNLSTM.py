"""
CNN, LSTM, and CNN-LSTM models from Thwe et al. (2026).

Reference:
    Y. Thwe, D. Maneetham, and P. N. Crisnapati,
    "Integrating EEG Artifact Removal with Deep Learning for Accurate Motor Imagery
    Classification in Acute Stroke,"
    Engineering, Technology & Applied Science Research, vol. 16, no. 1, 2026.

Architecture descriptions from the paper:
- CNN: 1D Conv(64 filters, kernel=3, ReLU) -> MaxPool -> Flatten -> Dense(50, ReLU)
       -> Softmax output.
- LSTM: LSTM layer with 50 units (ReLU) -> Softmax output.
- CNN-LSTM: Combines the CNN spatial feature extractor with the LSTM temporal
            model. The CNN output (after MaxPool) is reshaped into a sequence
            and fed into the LSTM(50).

Note:
    The paper explicitly mentions a softmax output layer. However, the project
    trainer uses nn.CrossEntropyLoss (which internally applies log-softmax).
    Therefore these implementations output raw logits for compatibility with
    the existing training loop. If you need exact softmax outputs for
    inference/visualisation, you can call torch.softmax on the returned logits.
"""

import math
import torch
import torch.nn as nn


class ThweCNN(nn.Module):
    """
    Simple 1D-CNN classifier as described in Thwe et al. (2026).
    """

    def __init__(self,
                 num_channels: int,
                 num_classes: int,
                 num_bands: int,
                 input_length: int) -> None:
        super().__init__()

        total_channels = num_channels * num_bands

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=total_channels, out_channels=64,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        conv_out_length = input_length // 2
        self.flatten = nn.Flatten()
        self.dense = nn.Sequential(
            nn.Linear(64 * conv_out_length, 50),
            nn.ReLU(),
            nn.Linear(50, num_classes),
            # Softmax intentionally omitted; trainer uses CrossEntropyLoss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, bands, channels, time).
        Returns:
            Logits of shape (batch, num_classes).
        """
        batch_size, B, C, T = x.shape
        x = x.reshape(batch_size, B * C, T)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


class ThweLSTM(nn.Module):
    """
    Simple LSTM classifier as described in Thwe et al. (2026).
    """

    def __init__(self,
                 num_channels: int,
                 num_classes: int,
                 num_bands: int,
                 input_length: int) -> None:
        super().__init__()

        self.input_size = num_channels * num_bands

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=50,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(50, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, bands, channels, time).
        Returns:
            Logits of shape (batch, num_classes).
        """
        batch_size, B, C, T = x.shape
        # Reshape to (batch, time, features) for LSTM
        x = x.reshape(batch_size, B * C, T).permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        # Take the last time step
        x = lstm_out[:, -1, :]
        x = self.classifier(x)
        return x


class ThweCNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM classifier as described in Thwe et al. (2026).

    The CNN consists of a single 1D convolutional layer (64 filters, kernel=3,
    ReLU) followed by max-pooling. The pooled feature maps are reshaped into
    a temporal sequence and fed into an LSTM with 50 hidden units.
    """

    def __init__(self,
                 num_channels: int,
                 num_classes: int,
                 num_bands: int,
                 input_length: int) -> None:
        super().__init__()

        total_channels = num_channels * num_bands

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=total_channels, out_channels=64,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # After MaxPool1d(2), time dimension is halved
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=50,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(50, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, bands, channels, time).
        Returns:
            Logits of shape (batch, num_classes).
        """
        batch_size, B, C, T = x.shape
        x = x.reshape(batch_size, B * C, T)

        # CNN feature extraction: (batch, 64, T//2)
        x = self.cnn(x)

        # Reshape for LSTM: (batch, T//2, 64)
        x = x.permute(0, 2, 1)

        # LSTM temporal modelling
        lstm_out, _ = self.lstm(x)

        # Take the final time step
        x = lstm_out[:, -1, :]
        x = self.classifier(x)
        return x
