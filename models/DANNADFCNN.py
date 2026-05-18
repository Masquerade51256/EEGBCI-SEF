"""
Domain-Adversarial Neural Network (DANN) based on ADFCNN backbone.

Architecture:
    Input -> ADFCNN Backbone -> Features -> [Classifier (task)]
                                         -> [Gradient Reversal -> Domain Discriminator]

Reference:
    Ganin et al. "Domain-Adversarial Training of Neural Networks", JMLR 2016
"""

import math
from typing import List

import torch
import torch.nn as nn
from models.myADFCNN import backbone, classifier


class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL).

    Forward: identity mapping
    Backward: reverse gradients and scale by lambda
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class DomainDiscriminator(nn.Module):
    """
    Domain discriminator network with hidden layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        num_domains: int = 4,
        dropout: float = 0.5,
    ):
        super(DomainDiscriminator, self).__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, num_domains))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DANNADFCNN(nn.Module):
    """
    Domain-Adversarial ADFCNN for cross-subject EEG-BCI.

    During training:
        outputs = model(x, alpha)
        class_output, domain_output = outputs
        # class_output: [B, num_classes]
        # domain_output: [B, num_domains]

    During inference:
        class_output = model(x)  # or model(x, alpha=None)
    """

    def __init__(
        self,
        num_classes: int,
        num_channels: int,
        num_bands: int,
        input_length: int,
        num_domains: int = 4,
        domain_hidden_dims: List[int] = None,
        use_band_attention: bool = True,
    ):
        super(DANNADFCNN, self).__init__()

        self.num_bands = num_bands
        self.num_channels = num_channels

        # Backbone expects [B, 1, C, T] where C is the actual spatial channel count.
        # When num_bands > 1, we reshape input from [B, num_bands, C, T] to
        # [B, 1, num_bands*C, T] in forward(), matching the ADFCNNAdapter logic.
        self.backbone = backbone(
            num_channels=num_channels * num_bands,
            input_timepoints=input_length,
        )

        # Dynamically infer feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, num_bands, num_channels, input_length)
            dummy_features = self.backbone(
                dummy_input.reshape(1, 1, num_channels * num_bands, -1)
            )
            flattened_features = dummy_features.shape[1]

        # Task classifier
        self.classifier = classifier(
            input_features=flattened_features, num_classes=num_classes
        )

        # Domain discriminator
        self.domain_discriminator = DomainDiscriminator(
            input_dim=flattened_features,
            hidden_dims=domain_hidden_dims,
            num_domains=num_domains,
        )

    def forward(self, x, alpha: float = 1.0):
        """
        Forward pass.

        Args:
            x: Input tensor [B, num_bands, C, T]
            alpha: Gradient reversal scaling factor.
                   If None, only return classification output (inference mode).

        Returns:
            class_output only if alpha is None or not training
            (class_output, domain_output) if training and alpha is not None
        """
        B = x.shape[0]
        # Reshape to match backbone expectation: [B, 1, num_bands*C, T]
        x = x.reshape(B, 1, self.num_channels * self.num_bands, -1)

        features = self.backbone(x)
        class_output = self.classifier(features)

        if self.training and alpha is not None:
            reversed_features = GradientReversalLayer.apply(features, alpha)
            domain_output = self.domain_discriminator(reversed_features)
            return class_output, domain_output

        return class_output
