"""
GACLNet.py
PyTorch implementation of the GACL-Net model, adapted to fit the existing BCI experimental platform.
Interface is standardized to match other models in the project (e.g., ADFCNN, EEGNet).
Author: [Your Name]
Reference: Original GACL-Net design and the structure of ADFCNN.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GACL_Backbone(nn.Module):
    """
    Backbone of the GACL-Net, responsible for feature extraction.
    This module is designed to handle input shaped as (batch, num_bands, num_channels, timepoints).
    """
    def __init__(self, num_channels: int, num_bands: int, dropout_rate=0.3):
        """
        Args:
            num_channels: Number of EEG channels.
            num_bands: Number of frequency bands (input channels in the spectral dimension).
            dropout_rate: Dropout probability.
        """
        super(GACL_Backbone, self).__init__()
        # The input has `num_bands` as the channel dimension for the first conv layer.
        # We treat the spatial (num_channels) and temporal dimensions jointly as the sequence length
        # after an initial reshaping.

        # 1. Initial projection to adapt to GACL-Net's expected input shape
        # We first reduce the (bands x channels) grid into a 1D sequence.
        # A simple approach: use a 1x1 conv to project bands to 1 channel, keeping spatial-temporal structure.
        self.input_projection = nn.Conv2d(num_bands, 1, kernel_size=(1, 1))
        
        # After projection, the data is shaped: (batch, 1, num_channels, timepoints)
        # We will flatten the (num_channels, timepoints) into a 1D sequence in the forward pass.

        # 2. Multi-scale Convolutional Block (adapted for 1D sequence)
        self.multiscale_conv = _MultiScaleConvBlock(in_channels=1, dropout_rate=dropout_rate)
        
        # 3. Attention Fusion Layer
        self.attention_fusion = _AttentionFusion(in_channels=96)  # 96 is the output channels of MultiScaleConvBlock
        
        # 4. Graph Convolutional Layer (implemented as 1D Conv)
        self.graph_conv = _GraphConvLayer(in_channels=96, dropout_rate=dropout_rate)
        
        # 5. Bi-directional LSTM with Attention
        self.bilstm_attention = _BiLSTMWithAttention(in_channels=64, dropout_rate=dropout_rate)  # 64 is the output of graph_conv
        
        # 6. Flatten layer
        self.flatten = nn.Flatten()
        
        # A placeholder for the feature dimension, will be set after a dummy forward pass
        self._output_features = None

    def forward(self, x):
        """
        Forward pass of the backbone.
        Args:
            x: Input tensor of shape (batch_size, num_bands, num_channels, timepoints).
        Returns:
            features: Flattened feature tensor of shape (batch_size, output_features).
        """
        # x shape: (B, C_band, C_channel, T)
        B, C_band, C_channel, T = x.shape
        
        # Step 1: Project frequency bands to a single channel
        x = self.input_projection(x)  # Output: (B, 1, C_channel, T)
        
        # Step 2: Reshape to a 1D sequence: merge spatial and temporal dimensions
        x = x.view(B, 1, C_channel * T)  # New shape: (B, 1, C_channel * T)
        
        # Step 3-6: Pass through the GACL-Net components
        x = self.multiscale_conv(x)
        x = self.attention_fusion(x)
        x = self.graph_conv(x)
        x = self.bilstm_attention(x)  # Output: (B, 128)
        
        # Flatten (though it's already 2D, this ensures consistency)
        features = self.flatten(x)  # Shape: (B, 128)
        
        # Dynamically record the output feature dimension
        if self._output_features is None:
            self._output_features = features.shape[1]
            
        return features


class GACL_Classifier(nn.Module):
    """Classifier head for GACL-Net."""
    def __init__(self, input_features: int, num_classes: int, dropout_rate=0.3):
        super(GACL_Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


class GACLNet(nn.Module):
    """
    Main GACL-Net model compatible with the existing BCI platform.
    Interface matches ADFCNN: (num_classes, num_channels, num_bands, input_length).
    """
    def __init__(self,
                 num_classes: int,
                 num_channels: int,
                 num_bands: int,
                 input_length: int):
        """
        Args:
            num_classes: Number of target classes.
            num_channels: Number of EEG electrodes/channels.
            num_bands: Number of frequency bands (input channels).
            input_length: Number of time samples per trial.
        """
        super(GACLNet, self).__init__()
        
        # Initialize backbone
        self.backbone = GACL_Backbone(num_channels=num_channels, num_bands=num_bands)
        
        # Dynamically determine the feature dimension from the backbone
        with torch.no_grad():
            # Create a dummy input matching the expected platform format: (bands, channels, time)
            dummy_input = torch.randn(1, num_bands, num_channels, input_length)
            dummy_features = self.backbone(dummy_input)
            flattened_features = dummy_features.shape[1]  # This should be 128 given our design
        
        # Initialize classifier with the determined feature dimension
        self.classifier = GACL_Classifier(input_features=flattened_features, num_classes=num_classes)
        
    def forward(self, x):
        """
        Forward pass of the full GACLNet.
        Args:
            x: Input tensor of shape (batch_size, num_bands, num_channels, input_length).
        Returns:
            logits: Class logits of shape (batch_size, num_classes).
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


# ==================== Sub-module Definitions (can be placed in a separate layers.py file) ====================

class _MultiScaleConvBlock(nn.Module):
    """Multi-scale 1D Convolutional Block."""
    def __init__(self, in_channels=1, out_channels=32, dropout_rate=0.3):
        super(_MultiScaleConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)
        self.batch_norm = nn.BatchNorm1d(out_channels * 3)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        conv1_out = self.relu(self.conv1(x))
        conv2_out = self.relu(self.conv2(x))
        conv3_out = self.relu(self.conv3(x))
        concat_out = torch.cat([conv1_out, conv2_out, conv3_out], dim=1)
        out = self.batch_norm(concat_out)
        out = self.dropout(out)
        return out


class _AttentionFusion(nn.Module):
    """Simplified Attention Fusion Layer."""
    def __init__(self, in_channels, reduction=8):
        super(_AttentionFusion, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )
        self.batch_norm = nn.BatchNorm1d(in_channels)

    def forward(self, x):
        batch_size, channels, seq_len = x.size()
        squeezed = F.avg_pool1d(x, seq_len).view(batch_size, channels)
        attention_weights = self.attention(squeezed).view(batch_size, channels, 1)
        attended = x * attention_weights.expand_as(x)
        out = x + attended
        out = self.batch_norm(out)
        return out


class _GraphConvLayer(nn.Module):
    """Graph Convolutional Layer implemented as a standard 1D convolution."""
    def __init__(self, in_channels, out_channels=64, kernel_size=3, dropout_rate=0.3):
        super(_GraphConvLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv(x))
        out = self.batch_norm(out)
        out = self.dropout(out)
        return out


class _BiLSTMWithAttention(nn.Module):
    """Bi-directional LSTM with a simplified attention mechanism."""
    def __init__(self, in_channels, hidden_size=64, num_layers=1, dropout_rate=0.3):
        super(_BiLSTMWithAttention, self).__init__()
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x_permuted = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x_permuted)
        attention_scores = torch.tanh(self.attention(lstm_out))
        attention_weights = F.softmax(attention_scores, dim=1)
        attended = torch.sum(lstm_out * attention_weights, dim=1)
        attended = self.dropout(attended)
        return attended


if __name__ == "__main__":
    # Unit test to verify the interface matches the framework's expectations
    num_classes = 2
    num_channels = 22
    num_bands = 4
    input_length = 1000
    
    model = GACLNet(num_classes=num_classes,
                    num_channels=num_channels,
                    num_bands=num_bands,
                    input_length=input_length)
    
    # Test with a dummy batch
    batch_size = 8
    dummy_input = torch.randn(batch_size, num_bands, num_channels, input_length)
    output = model(dummy_input)
    
    print(f"Model created successfully.")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, {num_classes})")
    assert output.shape == (batch_size, num_classes), f"Output shape mismatch: {output.shape}"
    print("Interface compatibility test passed.")