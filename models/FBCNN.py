import torch
import torch.nn as nn

class FilterBankCNN(nn.Module):
    """
    A CNN that processes multiple filter banks of EEG signals in parallel.

    Expected input shape: (batch_size, n_filterbanks, n_channels, n_times)
    Output shape: (batch_size, n_classes)

    Args:
        n_filterbanks (int): Number of filter banks (frequency bands).
        n_channels (int): Number of EEG channels.
        n_times (int): Number of time samples per trial.
        n_classes (int): Number of target classes.
        temporal_kernel_length (int): Length of the temporal convolution kernel in the first layer. Default: 64.
        spatial_kernel_size (int): Size of the spatial (depthwise) convolution kernel. Should be (n_channels, 1). We use 1 for pointwise spatial conv here for simplicity. A more complex design could use (n_channels, 1).
        n_filters_first (int): Number of filters in the first temporal convolution. Default: 8.
        pool_mode (str): Pooling mode, 'mean' for AvgPool2d or 'max' for MaxPool2d. Default: 'mean'.
        pool_size (int): Kernel size for the temporal pooling. Default: 4.
        dropout_rate (float): Dropout rate applied after pooling. Default: 0.5.
        depth_multiplier (int, optional): Depth multiplier for depthwise separable convolutions. If > 1, enables a more complex block. Default: 1 (standard conv).
    """

    def __init__(self,
                 n_filterbanks: int,
                 n_channels: int,
                 n_times: int,
                 n_classes: int,
                 temporal_kernel_length: int = 64,
                 spatial_kernel_size: int = 1, # Using 1 for simplicity. For true spatial filter, set to n_channels.
                 n_filters_first: int = 8,
                 pool_mode: str = 'mean',
                 pool_size: int = 4,
                 dropout_rate: float = 0.5,
                 depth_multiplier: int = 1):
        super(FilterBankCNN, self).__init__()

        self.n_filterbanks = n_filterbanks
        self.n_channels = n_channels
        self.n_times = n_times

        # Define the shared feature extractor for each filter bank
        # This block will be applied to each (channel, time) slice of a single filterbank
        shared_layers = []
        
        # Layer 1: Temporal Convolution (learns band-specific temporal filters)
        # Input shape per band: (1, n_channels, n_times) after unsqueeze(1)
        # Output: (n_filters_first, n_channels, n_times')
        padding = (temporal_kernel_length // 2, 0) # Pad only in time dimension
        shared_layers.append(
            nn.Conv2d(in_channels=1, # Each filterbank is processed independently
                      out_channels=n_filters_first,
                      kernel_size=(1, temporal_kernel_length), # (spatial=1, temporal)
                      padding=padding,
                      bias=False)
        )
        shared_layers.append(nn.BatchNorm2d(n_filters_first))
        shared_layers.append(nn.ELU())

        # Layer 2: Spatial Convolution (Depthwise or Pointwise across channels)
        # Using a simple pointwise (1x1) conv here for channel mixing.
        # For a true spatial filter as in EEGNet, you would set kernel_size=(n_channels, 1) and groups=n_filters_first.
        spatial_out_channels = n_filters_first * depth_multiplier
        shared_layers.append(
            nn.Conv2d(in_channels=n_filters_first,
                      out_channels=spatial_out_channels,
                      kernel_size=(spatial_kernel_size, 1), # (spatial, 1)
                      groups=n_filters_first if spatial_kernel_size == n_channels else 1, # Depthwise if full spatial kernel
                      bias=False)
        )
        shared_layers.append(nn.BatchNorm2d(spatial_out_channels))
        shared_layers.append(nn.ELU())

        # Pooling and Dropout
        PoolLayer = nn.AvgPool2d if pool_mode == 'mean' else nn.MaxPool2d
        shared_layers.append(PoolLayer(kernel_size=(1, pool_size)))
        shared_layers.append(nn.Dropout(p=dropout_rate))

        # This sequential module will be applied to each filter bank
        self.shared_feature_extractor = nn.Sequential(*shared_layers)

        # ------ Calculate the size of features after the shared extractor ------
        # We run a dummy input through the shared extractor to get the output feature dimensions
        with torch.no_grad():
            dummy_input_single_band = torch.randn(1, 1, n_channels, n_times)
            dummy_output_single_band = self.shared_feature_extractor(dummy_input_single_band)
            self.features_per_band = dummy_output_single_band.numel() # Flattened size for one band

        # Total features after concatenating all bands
        total_features = self.features_per_band * n_filterbanks

        # Final Classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features=total_features, out_features=n_classes, bias=True),
            # Note: No softmax here. Use with nn.CrossEntropyLoss which includes LogSoftmax.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Filter Bank CNN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_filterbanks, n_channels, n_times).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, n_classes).
        """
        batch_size = x.shape[0]
        # We will process each filter bank and collect the features
        band_features = []

        for band_idx in range(self.n_filterbanks):
            # Select the data for the current filter bank: shape (batch, 1, channels, times)
            band_data = x[:, band_idx:band_idx+1, :, :] # Keep the channel dimension as 1 for conv2d
            
            # Apply the shared feature extractor
            extracted = self.shared_feature_extractor(band_data) # shape: (batch, C, H, W)
            
            # Flatten the extracted features for this band
            flattened = extracted.view(batch_size, -1) # shape: (batch, features_per_band)
            band_features.append(flattened)

        # Concatenate features from all bands along the feature dimension
        combined_features = torch.cat(band_features, dim=1) # shape: (batch, total_features)

        # Pass through the final classifier
        output = self.classifier(combined_features) # shape: (batch, n_classes)
        return output