"""
XWFilterBankNet: A Modular Filter Bank Deep Learning Model for EEG Classification

This model is designed for stroke patient EEG datasets (XWStroke) but is 
modular enough to work with other EEG datasets like BCICIV2a.

Architecture:
    1. Multi-band Feature Extraction (per-filter-bank convolutions)
    2. Cross-band Attention Fusion (learn band importance)
    3. Spatio-Temporal Feature Extraction
    4. Global Temporal Pooling
    5. Classifier

Input Shape: (batch, n_bands, n_channels, n_times)
Output Shape: (batch, n_classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math

from models.layers import Conv2dWithConstraint, LinearWithConstraint


class BandFeatureExtractor(nn.Module):
    """
    Feature extractor for a single frequency band.
    
    Uses depthwise separable convolutions to extract spatial and temporal
    features efficiently from a single band.
    
    Args:
        in_channels: Number of input channels (EEG channels)
        out_channels: Number of output feature channels
        kernel_temporal: Temporal convolution kernel size
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 32,
        kernel_temporal: int = 64,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_temporal = kernel_temporal
        
        # Temporal convolution (band-specific temporal patterns)
        # Use odd kernel size to avoid padding warning with 'same' padding
        effective_kernel = kernel_temporal if kernel_temporal % 2 == 1 else kernel_temporal + 1
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, out_channels, (1, effective_kernel), 
                     padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU()
        )
        
        # Spatial convolution (depthwise across channels)
        self.spatial_conv = nn.Sequential(
            Conv2dWithConstraint(
                out_channels, out_channels,
                (in_channels, 1),
                groups=out_channels,
                bias=False,
                max_norm=1.0
            ),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            nn.Dropout2d(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for single band.
        
        Args:
            x: Input of shape (batch, 1, channels, times)
        
        Returns:
            Features of shape (batch, out_channels, 1, times)
        """
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        return x


class CrossBandAttention(nn.Module):
    """
    Cross-band attention module to learn importance of different frequency bands.
    
    This allows the model to focus on bands that are most discriminative for
    the classification task (e.g., mu rhythm for motor imagery).
    
    Args:
        n_bands: Number of frequency bands
        feature_dim: Dimension of features per band
    """
    
    def __init__(self, n_bands: int, feature_dim: int):
        super().__init__()
        
        self.n_bands = n_bands
        self.feature_dim = feature_dim
        
        # Attention weights computation
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, max(feature_dim // 4, 1)),
            nn.Tanh(),
            nn.Linear(max(feature_dim // 4, 1), 1)
        )
        
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention across bands.
        
        Args:
            x: Input of shape (batch, n_bands, feature_dim)
        
        Returns:
            Attended features of shape (batch, n_bands, feature_dim)
            Attention weights of shape (batch, n_bands, 1)
        """
        # Compute attention scores
        attn_scores = self.attention(x)  # (batch, n_bands, 1)
        attn_weights = self.softmax(attn_scores)  # Normalize across bands
        
        # Apply attention
        attended = x * attn_weights  # Element-wise multiplication
        
        return attended, attn_weights


class XWFilterBankNet(nn.Module):
    """
    XWFilterBankNet: Modular Filter Bank Network for EEG Classification
    
    This model is designed to work with any EEG dataset that uses filter bank
    preprocessing. It extracts features from multiple frequency bands and uses
    attention mechanisms to focus on discriminative bands.
    
    Architecture:
        1. Per-band feature extraction (parallel convolutions for each band)
        2. Cross-band attention (learn which bands are important)
        3. Band fusion (max pooling across bands)
        4. Temporal aggregation (global average pooling)
        5. Classification
    
    Args:
        num_classes: Number of output classes
        num_channels: Number of EEG channels
        num_bands: Number of frequency bands (from filter bank)
        input_length: Number of time samples
        band_channels: Base number of channels for band feature extraction
        temporal_kernel: Kernel size for temporal convolutions
        dropout: Dropout rate
        use_attention: Whether to use cross-band attention
        classifier_hidden: Hidden units in classifier (0 = linear)
    
    Example:
        >>> model = XWFilterBankNet(
        ...     num_classes=2,
        ...     num_channels=30,
        ...     num_bands=9,
        ...     input_length=500
        ... )
    """
    
    def __init__(
        self,
        num_classes: int,
        num_channels: int,
        num_bands: int,
        input_length: int,
        band_channels: int = 32,
        temporal_kernel: int = 64,
        dropout: float = 0.3,
        use_attention: bool = True,
        classifier_hidden: int = 0
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.num_bands = num_bands
        self.input_length = input_length
        self.band_channels = band_channels
        self.use_attention = use_attention
        
        # Per-band feature extractors (shared weights across bands)
        self.band_extractor = BandFeatureExtractor(
            in_channels=num_channels,
            out_channels=band_channels,
            kernel_temporal=temporal_kernel,
            dropout=dropout
        )
        
        # Feature dimension after band extraction: band_channels * input_length
        self.band_feature_dim = band_channels * input_length
        
        # Cross-band attention
        if use_attention:
            self.cross_band_attn = CrossBandAttention(
                n_bands=num_bands,
                feature_dim=self.band_feature_dim
            )
        
        # Temporal convolution after band fusion
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(band_channels, band_channels * 2, (1, temporal_kernel // 2), 
                     padding=(0, temporal_kernel // 4), bias=False),
            nn.BatchNorm2d(band_channels * 2),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout2d(dropout)
        )
        
        # Calculate output feature dimension
        # After temporal conv: (batch, band_channels*2, 1, time//4)
        self.temporal_out_length = input_length // 4
        self.feature_dim = band_channels * 2 * self.temporal_out_length
        
        # Classifier
        if classifier_hidden > 0:
            self.classifier = nn.Sequential(
                LinearWithConstraint(
                    self.feature_dim,
                    classifier_hidden,
                    max_norm=0.5
                ),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden, num_classes)
            )
        else:
            self.classifier = LinearWithConstraint(
                self.feature_dim,
                num_classes,
                max_norm=0.5
            )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, num_bands, num_channels, input_length)
        
        Returns:
            Output logits of shape (batch, num_classes)
        """
        batch_size = x.shape[0]
        
        # Process each band with shared weights
        band_features = []
        for i in range(self.num_bands):
            band_data = x[:, i:i+1, :, :]  # (batch, 1, channels, times)
            feat = self.band_extractor(band_data)  # (batch, band_channels, 1, times)
            # Flatten spatial dim: (batch, band_channels, times)
            feat = feat.squeeze(2)
            # Flatten features: (batch, band_channels * times)
            feat = feat.view(batch_size, -1)
            band_features.append(feat)
        
        # Stack band features: (batch, num_bands, band_feature_dim)
        stacked = torch.stack(band_features, dim=1)
        
        # Apply cross-band attention if enabled
        if self.use_attention:
            attended, _ = self.cross_band_attn(stacked)
            # Reshape back: (batch, num_bands, band_channels, times)
            stacked = attended.view(batch_size, self.num_bands, self.band_channels, self.input_length)
        else:
            # Reshape without attention
            stacked = stacked.view(batch_size, self.num_bands, self.band_channels, self.input_length)
        
        # Fuse bands using max pooling
        fused = torch.max(stacked, dim=1)[0]  # (batch, band_channels, times)
        fused = fused.unsqueeze(2)  # (batch, band_channels, 1, times)
        
        # Temporal convolution
        features = self.temporal_conv(fused)  # (batch, band_channels*2, 1, times//4)
        
        # Flatten and classify
        features = features.view(batch_size, -1)
        output = self.classifier(features)
        
        return output
    
    def get_band_attention(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Get attention weights for each frequency band.
        
        This can be used for interpretability to understand which
        frequency bands the model focuses on.
        
        Args:
            x: Input tensor of shape (batch, num_bands, num_channels, input_length)
        
        Returns:
            Attention weights of shape (batch, num_bands, 1) or None
        """
        if not self.use_attention:
            return None
        
        self.eval()
        with torch.no_grad():
            batch_size = x.shape[0]
            
            # Extract band features
            band_features = []
            for i in range(self.num_bands):
                band_data = x[:, i:i+1, :, :]
                feat = self.band_extractor(band_data)
                feat = feat.squeeze(2)
                feat = feat.view(batch_size, -1)
                band_features.append(feat)
            
            stacked = torch.stack(band_features, dim=1)
            
            # Get attention weights
            _, attn_weights = self.cross_band_attn(stacked)
            
        return attn_weights


class XWFilterBankNetLight(nn.Module):
    """
    Lightweight version of XWFilterBankNet with fewer parameters.
    
    Suitable for smaller datasets or when faster training is needed.
    
    Args:
        num_classes: Number of output classes
        num_channels: Number of EEG channels
        num_bands: Number of frequency bands
        input_length: Number of time samples
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        num_classes: int,
        num_channels: int,
        num_bands: int,
        input_length: int,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.num_bands = num_bands
        self.input_length = input_length
        
        # Simplified: shared feature extractor with fewer channels
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, 16, (1, 32), padding=(0, 16), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU()
        )
        
        self.spatial_conv = nn.Sequential(
            Conv2dWithConstraint(
                16, 16, (num_channels, 1),
                groups=16, bias=False, max_norm=1.0
            ),
            nn.BatchNorm2d(16),
            nn.ELU()
        )
        
        self.temporal_agg = nn.Sequential(
            nn.AvgPool2d((1, 4)),
            nn.Dropout2d(dropout)
        )
        
        # Output dimension calculation
        # After pooling: time // 4
        self.feature_dim = 16 * (input_length // 4)
        
        self.classifier = LinearWithConstraint(
            self.feature_dim, num_classes, max_norm=0.5
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input of shape (batch, num_bands, num_channels, input_length)
        
        Returns:
            Output of shape (batch, num_classes)
        """
        batch_size = x.shape[0]
        
        # Process each band
        band_feats = []
        for i in range(self.num_bands):
            band = x[:, i:i+1, :, :]
            feat = self.temporal_conv(band)
            feat = self.spatial_conv(feat)
            feat = self.temporal_agg(feat)
            feat = feat.view(batch_size, -1)
            band_feats.append(feat)
        
        # Max pooling across bands
        stacked = torch.stack(band_feats, dim=1)  # (batch, num_bands, features)
        fused = torch.max(stacked, dim=1)[0]  # (batch, features)
        
        # Classification
        return self.classifier(fused)


# Factory function for easy model creation
def create_xw_filterbanknet(
    variant: str = "standard",
    **kwargs
) -> nn.Module:
    """
    Factory function to create XWFilterBankNet variants.
    
    Args:
        variant: Model variant ("standard", "light", "deep")
        **kwargs: Model parameters
    
    Returns:
        Model instance
    """
    if variant == "standard":
        return XWFilterBankNet(**kwargs)
    elif variant == "light":
        return XWFilterBankNetLight(**kwargs)
    elif variant == "deep":
        # Deep variant with more layers
        kwargs.update({
            'band_channels': 64,
            'temporal_kernel': 128,
            'classifier_hidden': 128
        })
        return XWFilterBankNet(**kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant}")
