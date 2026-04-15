"""
Model definitions for EEG-BCI experiments.
"""

import torch.nn as nn
from core.registry import MODELS


def register_all_models():
    """
    Register all available models.
    
    This function should be called at startup to populate the model registry.
    """
    
    # EEGNet
    try:
        from .EEGNet import EEGNet
        MODELS.register('EEGNet')(EEGNet)
    except ImportError as e:
        print(f"Warning: Could not register EEGNet: {e}")
    
    # CNNLSTM variants
    try:
        from .CNNLSTM import CNNLSTM, MultiBand_CNNLSTM, Simplified_MultiBand_CNNLSTM
        MODELS.register('CNNLSTM')(CNNLSTM)
        MODELS.register('MultiBand_CNNLSTM')(MultiBand_CNNLSTM)
        MODELS.register('Simplified_MultiBand_CNNLSTM')(Simplified_MultiBand_CNNLSTM)
    except ImportError as e:
        print(f"Warning: Could not register CNNLSTM models: {e}")
    
    # ADFCNN
    try:
        from .myADFCNN import ADFCNN
        
        # Create adapter to handle multi-band input
        class ADFCNNAdapter(nn.Module):
            def __init__(self, num_channels, num_classes, num_bands, input_length, **kwargs):
                super().__init__()
                self.num_bands = num_bands
                self.num_channels = num_channels
                
                # ADFCNN expects input shape [B, 1, C, T]
                # But data comes as [B, num_bands, C, T]
                # We reshape to [B, 1, num_bands*C, T]
                self.model = ADFCNN(
                    num_classes=num_classes,
                    num_channels=num_channels * num_bands,  # Merge bands and channels
                    num_bands=1,  # After reshape, we have 1 "band"
                    input_length=input_length
                )
            
            def forward(self, x):
                # x: [B, num_bands, C, T]
                B = x.shape[0]
                # Reshape to [B, 1, num_bands*C, T]
                # Use reshape instead of view to handle non-contiguous tensors
                x = x.reshape(B, 1, self.num_channels * self.num_bands, -1)
                return self.model(x)
        
        MODELS.register('ADFCNN')(ADFCNNAdapter)
    except ImportError as e:
        print(f"Warning: Could not register ADFCNN: {e}")
    
    # FilterBankCNN
    try:
        from .FBCNN import FilterBankCNN
        
        # Create adapter to handle different parameter names
        class FilterBankCNNAdapter(nn.Module):
            def __init__(self, num_channels, num_classes, num_bands, input_length, **kwargs):
                super().__init__()
                # FilterBankCNN uses: n_filterbanks, n_channels, n_times, n_classes
                self.model = FilterBankCNN(
                    n_filterbanks=num_bands,
                    n_channels=num_channels,
                    n_times=input_length,
                    n_classes=num_classes
                )
            
            def forward(self, x):
                return self.model(x)
        
        MODELS.register('FilterBankCNN')(FilterBankCNNAdapter)
    except ImportError as e:
        print(f"Warning: Could not register FilterBankCNN: {e}")
    
    # FBCNet_Standard
    try:
        from .myFBCNet import FBCNet_Standard
        
        # Create adapter to handle different parameter names
        class FBCNetAdapter(nn.Module):
            def __init__(self, num_channels, num_classes, num_bands, input_length, **kwargs):
                super().__init__()
                # FBCNet_Standard uses: num_classes, num_channels, input_time_length, n_band
                self.model = FBCNet_Standard(
                    num_classes=num_classes,
                    num_channels=num_channels,
                    input_time_length=input_length,
                    n_band=num_bands
                )
            
            def forward(self, x):
                return self.model(x)
        
        MODELS.register('FBCNet_Standard')(FBCNetAdapter)
    except ImportError as e:
        print(f"Warning: Could not register FBCNet_Standard: {e}")
    
    # GACLNet
    try:
        from .GACLNet import GACLNet
        MODELS.register('GACLNet')(GACLNet)
    except ImportError as e:
        print(f"Warning: Could not register GACLNet: {e}")
    
    # XWFilterBankNet
    try:
        from .XWFilterBankNet import XWFilterBankNet, XWFilterBankNetLight
        MODELS.register('XWFilterBankNet')(XWFilterBankNet)
        MODELS.register('XWFilterBankNetLight')(XWFilterBankNetLight)
    except ImportError as e:
        print(f"Warning: Could not register XWFilterBankNet: {e}")

    # Thwe et al. (2026) models
    try:
        from .ThweCNNLSTM import ThweCNN, ThweLSTM, ThweCNNLSTM
        MODELS.register('ThweCNN')(ThweCNN)
        MODELS.register('ThweLSTM')(ThweLSTM)
        MODELS.register('ThweCNNLSTM')(ThweCNNLSTM)
    except ImportError as e:
        print(f"Warning: Could not register ThweCNNLSTM models: {e}")


# Auto-register on import
register_all_models()

# Export registry for convenience
__all__ = ['MODELS', 'register_all_models']
