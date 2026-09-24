"""
Clinical-Anchored Contrastive Meta-Learning (CACML) Model.

Dual-tower architecture that jointly learns EEG features and clinical metadata
embeddings in a shared space, enabling clinical-guided transfer learning.

Uses StableEEGBackbone: a fixed, deterministic CNN without dynamic layer
creation or self-attention, avoiding the segfault issues in ADFCNNBackbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StableEEGBackbone(nn.Module):
    """
    Stable EEG feature extractor for CACML.

    Design principles:
    - All layers created statically in __init__ (no dynamic creation in forward)
    - No self-attention (avoids complexity on small cohorts)
    - Dual-branch: large-kernel (global/low-freq) + small-kernel (local/high-freq)
    - Inspired by EEGNet/ADFCNN but simplified for stability

    Input:  [B, 1, num_channels, timepoints]
    Output: [B, feature_dim]
    """

    def __init__(self, num_channels, input_length, F1=8, D=2, dropout=0.25):
        super().__init__()
        F2 = F1 * D

        # ---------- Branch I: large kernel (global, low-frequency) ----------
        self.temporal_b1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, 125), padding='same', bias=False),
            nn.BatchNorm2d(F1),
        )
        self.spatial_b1 = nn.Sequential(
            # Depthwise: each feature map independently across channels
            nn.Conv2d(F1, F2, kernel_size=(num_channels, 1), padding=0,
                      groups=F1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.Dropout(dropout),
            # Pointwise: mix feature maps
            nn.Conv2d(F2, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 32), stride=(1, 32)),
            nn.Dropout(dropout),
        )

        # ---------- Branch II: small kernel (local, high-frequency) ----------
        self.temporal_b2 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, 30), padding='same', bias=False),
            nn.BatchNorm2d(F1),
        )
        self.spatial_b2 = nn.Sequential(
            # Standard convolution (no depthwise separation)
            nn.Conv2d(F1, F2, kernel_size=(num_channels, 1), padding=0, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 75), stride=(1, 25)),
            nn.Dropout(dropout),
        )

        self.flatten = nn.Flatten()

        # Pre-compute output dimension (after flatten)
        with torch.no_grad():
            dummy = torch.randn(1, 1, num_channels, input_length)
            out = self.forward(dummy)
            self.output_dim = out.shape[1]

    def _forward_core(self, x):
        """Core forward without flatten (for dim probing)."""
        x1 = self.temporal_b1(x)
        x1 = self.spatial_b1(x1)

        x2 = self.temporal_b2(x)
        x2 = self.spatial_b2(x2)

        # Concatenate along time dimension
        x_cat = torch.cat([x1, x2], dim=3)
        return x_cat

    def forward(self, x):
        x_cat = self._forward_core(x)
        return self.flatten(x_cat)


class CACML(nn.Module):
    """
    CACML: Clinical-Anchored Contrastive Meta-Learning.

    Architecture:
    - EEG Tower: StableEEGBackbone -> projection -> L2-normalized embedding
    - Clinical Tower: MLP(7 -> 32 -> 64 -> 128) -> L2-normalized embedding
    - Classifier: MLP(128 -> 64 -> num_classes)

    Args:
        num_classes: Number of output classes (default 2 for left/right MI)
        num_channels: Number of EEG channels (e.g., 32 for XWStroke)
        num_bands: Number of frequency bands (e.g., 4 for filter bank)
        input_length: Number of time points per trial
        embed_dim: Dimension of the joint embedding space (default 128)
        dropout: Dropout rate for classifier (default 0.5)
    """

    def __init__(self, num_classes, num_channels, num_bands, input_length,
                 embed_dim=128, dropout=0.5):
        super().__init__()
        self.num_channels = num_channels
        self.num_bands = num_bands
        self.embed_dim = embed_dim

        # EEG Tower: stable backbone (no dynamic layer creation)
        self.eeg_backbone = StableEEGBackbone(
            num_channels=num_channels * num_bands,
            input_length=input_length,
        )
        eeg_feature_dim = self.eeg_backbone.output_dim

        self.eeg_proj = nn.Sequential(
            nn.Linear(eeg_feature_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )

        # Clinical Tower
        # Input: [loc_encoding(3), NIHSS_norm, duration_norm, side_encoding(2)] = 7-dim
        self.clinical_tower = nn.Sequential(
            nn.Linear(7, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
        )

        # Classifier operates on EEG embeddings
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x_eeg, x_clinical=None, return_features=False):
        """
        Forward pass.

        Args:
            x_eeg: EEG input, shape [B, num_bands, C, T]
            x_clinical: Clinical features, shape [B, 7] or None
            return_features: If True, also return f_eeg and f_clin

        Returns:
            logits: [B, num_classes]
            (Optional) f_eeg: [B, embed_dim], f_clin: [B, embed_dim] or None
        """
        # EEG path: reshape multi-band -> single-band for backbone
        B = x_eeg.shape[0]
        x_eeg_reshaped = x_eeg.reshape(B, 1, self.num_channels * self.num_bands, -1)
        eeg_feat = self.eeg_backbone(x_eeg_reshaped)  # [B, eeg_feature_dim]
        f_eeg_raw = self.eeg_proj(eeg_feat)           # [B, embed_dim], unnormalized
        f_eeg = F.normalize(f_eeg_raw, p=2, dim=1)    # normalized for contrastive/align

        # Clinical path
        if x_clinical is not None:
            f_clin = self.clinical_tower(x_clinical)  # [B, embed_dim]
            f_clin = F.normalize(f_clin, p=2, dim=1)
        else:
            f_clin = None

        # Classification: use UNNORMALIZED projection for better class separability
        logits = self.classifier(f_eeg_raw)

        if return_features:
            return logits, f_eeg, f_clin
        return logits


class CACMLLoss(nn.Module):
    """
    Combined loss for CACML training.

    Components:
    1. Classification loss: standard cross-entropy on predicted labels
    2. Contrastive loss: configurable contrastive learning mode
       - clinical_anchor: original InfoNCE with clinical hard negatives (can collapse)
       - class_aware: supervised contrastive using MI labels only
       - clinical_informed: class-aware positives weighted by clinical similarity (Scheme 3)
    3. Alignment loss: direct L2 distance between f_eeg and f_clin for same sample

    Args:
        lambda_contrast: Weight for contrastive loss (default 1.0)
        lambda_align: Weight for alignment loss (default 0.5)
        lambda_cls: Weight for classification loss (default 1.0)
        tau: Temperature for EEG similarity in contrastive loss (default 0.07)
        tau_clin: Temperature for clinical similarity weighting (default 0.07)
        contrast_mode: One of 'clinical_anchor', 'class_aware', 'clinical_informed'
        use_hard_negative: Enable hard negative mining (clinical_anchor only)
        k_hard: Number of hard negatives per sample (clinical_anchor only)
        alpha_hard: Weight multiplier for hard negatives (clinical_anchor only)
    """

    def __init__(self, lambda_contrast=1.0, lambda_align=0.5, lambda_cls=1.0, tau=0.07,
                 tau_clin=0.07, contrast_mode='clinical_anchor',
                 use_hard_negative=True, k_hard=5, alpha_hard=2.0):
        super().__init__()
        self.lambda_contrast = lambda_contrast
        self.lambda_align = lambda_align
        self.lambda_cls = lambda_cls
        self.tau = tau
        self.tau_clin = tau_clin
        self.contrast_mode = contrast_mode
        self.use_hard_negative = use_hard_negative
        self.k_hard = k_hard
        self.alpha_hard = alpha_hard
        self.ce = nn.CrossEntropyLoss()

        if contrast_mode not in ('clinical_anchor', 'class_aware', 'clinical_informed'):
            raise ValueError(f"Unknown contrast_mode: {contrast_mode}")

    def forward(self, logits, labels, f_eeg, f_clin=None):
        """
        Compute combined CACML loss.

        Args:
            logits: [B, num_classes]
            labels: [B]
            f_eeg: [B, embed_dim], L2-normalized
            f_clin: [B, embed_dim], L2-normalized (required for clinical_* modes)

        Returns:
            total_loss (scalar tensor)
            loss_dict (dict of float values for logging)
        """
        # 1. Classification loss
        loss_cls = self.ce(logits, labels)

        # 2. Alignment loss: same-sample EEG and clinical should be close
        if f_clin is not None and self.lambda_align > 0:
            loss_align = F.mse_loss(f_eeg, f_clin)
        else:
            loss_align = torch.tensor(0.0, device=f_eeg.device)

        # 3. Contrastive loss
        if self.contrast_mode == 'clinical_anchor':
            if f_clin is None:
                raise ValueError("clinical_anchor mode requires f_clin")
            loss_contrast = self._clinical_anchor_contrastive(f_eeg, f_clin)
        elif self.contrast_mode == 'class_aware':
            loss_contrast = self._class_aware_contrastive(f_eeg, labels)
        elif self.contrast_mode == 'clinical_informed':
            if f_clin is None:
                raise ValueError("clinical_informed mode requires f_clin")
            loss_contrast = self._clinical_informed_contrastive(f_eeg, f_clin, labels)
        else:
            raise ValueError(f"Unknown contrast_mode: {self.contrast_mode}")

        # Total loss (lambda_cls boosts classifier learning when contrastive dominates)
        total = (self.lambda_cls * loss_cls
                 + self.lambda_contrast * loss_contrast
                 + self.lambda_align * loss_align)

        loss_dict = {
            'loss_cls': loss_cls.item(),
            'loss_contrast': loss_contrast.item(),
            'loss_align': loss_align.item(),
            'total': total.item(),
            'lambda_cls': self.lambda_cls,
            'contrast_mode': self.contrast_mode,
        }

        return total, loss_dict

    # ------------------------------------------------------------------
    # Contrastive loss implementations
    # ------------------------------------------------------------------

    def _clinical_anchor_contrastive(self, f_eeg, f_clin):
        """Original clinical-anchor InfoNCE (can pull different MI labels together)."""
        B = f_eeg.shape[0]
        device = f_eeg.device

        sim = torch.mm(f_eeg, f_clin.t()) / self.tau  # [B, B]

        if self.use_hard_negative and B > 1:
            clin_sim = torch.mm(f_clin, f_clin.t())
            mask = torch.eye(B, device=device)
            clin_sim_masked = clin_sim - mask * 1e9

            k = min(self.k_hard, B - 1)
            _, topk_idx = torch.topk(clin_sim_masked, k=k, dim=1)

            weight_mask = torch.ones_like(sim)
            for i in range(B):
                weight_mask[i, topk_idx[i]] = self.alpha_hard

            log_weights = torch.log(weight_mask)
            weighted_sim = sim + log_weights

            labels_idx = torch.arange(B, device=device)
            loss_e2c = F.cross_entropy(weighted_sim, labels_idx)
            loss_c2e = F.cross_entropy(weighted_sim.t(), labels_idx)
        else:
            labels_idx = torch.arange(B, device=device)
            loss_e2c = F.cross_entropy(sim, labels_idx)
            loss_c2e = F.cross_entropy(sim.t(), labels_idx)

        return (loss_e2c + loss_c2e) / 2.0

    def _class_aware_contrastive(self, f_eeg, labels):
        """Supervised contrastive: positives = same MI label (Scheme 1)."""
        f_eeg = F.normalize(f_eeg, p=2, dim=1)
        sim_eeg = torch.matmul(f_eeg, f_eeg.t()) / self.tau  # [B, B]

        labels = labels.view(-1, 1)
        mask_pos = torch.eq(labels, labels.t()).float()
        mask_pos.fill_diagonal_(0)

        mask_all = torch.ones_like(mask_pos)
        mask_all.fill_diagonal_(0)

        exp_sim = torch.exp(sim_eeg)
        numerator = (exp_sim * mask_pos).sum(dim=1)
        denominator = (exp_sim * mask_all).sum(dim=1)

        valid = (mask_pos.sum(dim=1) > 0).float()
        loss = -torch.log((numerator + 1e-8) / (denominator + 1e-8))
        return (loss * valid).sum() / (valid.sum() + 1e-8)

    def _clinical_informed_contrastive(self, f_eeg, f_clin, labels):
        """
        Scheme 3: class-aware positives weighted by clinical similarity.

        Positives are determined by MI label; clinical similarity only modulates
        how strongly each positive pair is pulled together.
        """
        f_eeg = F.normalize(f_eeg, p=2, dim=1)
        f_clin = F.normalize(f_clin, p=2, dim=1)

        # EEG similarity (contrastive signal)
        sim_eeg = torch.matmul(f_eeg, f_eeg.t()) / self.tau  # [B, B]

        # Clinical similarity (weighting signal only)
        sim_clin = torch.matmul(f_clin, f_clin.t()) / self.tau_clin  # [B, B]

        # Positive mask: same MI label, exclude self
        labels = labels.view(-1, 1)
        mask_pos = torch.eq(labels, labels.t()).float()
        mask_pos.fill_diagonal_(0)

        # All-other mask (positives + negatives), exclude self
        mask_all = torch.ones_like(mask_pos)
        mask_all.fill_diagonal_(0)

        exp_sim_eeg = torch.exp(sim_eeg)
        exp_sim_clin = torch.exp(sim_clin)

        # Numerator: same-label pairs weighted by clinical similarity
        numerator = (exp_sim_eeg * exp_sim_clin * mask_pos).sum(dim=1)
        # Denominator: all other samples (keeps negative-sample repulsion)
        denominator = (exp_sim_eeg * mask_all).sum(dim=1)

        valid = (mask_pos.sum(dim=1) > 0).float()
        loss = -torch.log((numerator + 1e-8) / (denominator + 1e-8))
        return (loss * valid).sum() / (valid.sum() + 1e-8)
