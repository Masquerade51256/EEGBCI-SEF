"""
Trainers for Clinical-Anchored Contrastive Meta-Learning (CACML).

Implements two training modes:
1. CACMLPretrainTrainer: Joint pretraining of EEG and Clinical towers on all source subjects
2. CACMLLOSOSTrainer: Leave-One-Subject-Out evaluation with optional n-shot fine-tuning
"""

import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from core.base.base_trainer import BaseTrainer
from core.registry import MODELS, build_from_config
from data.datasets import get_clinical_feature_vector, XWStrokeDataset
from models.CACML import CACMLLoss


# ---------------------------------------------------------------------------
# Helper dataset for pretraining: yields (data, label, clinical_features)
# ---------------------------------------------------------------------------

class CACMLPretrainDataset(Dataset):
    """
    Concatenates multiple subject datasets and attaches pre-loaded clinical features.

    Each sample returns: (eeg_tensor, label_tensor, clinical_tensor)
    """

    def __init__(self, subject_datasets: List[Dataset], dataset_info: Dict):
        self.subject_datasets = subject_datasets
        self.cumulative_lengths = [0]
        for ds in subject_datasets:
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(ds))
        self.total_len = self.cumulative_lengths[-1]

        # Pre-load clinical features for all subjects
        self.clinical_features = {}
        for ds in subject_datasets:
            self.clinical_features[ds.subject_id] = get_clinical_feature_vector(
                ds.subject_id, dataset_info
            )

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx: int):
        # Find which subject dataset this index belongs to
        for i in range(len(self.subject_datasets)):
            if idx < self.cumulative_lengths[i + 1]:
                local_idx = idx - self.cumulative_lengths[i]
                ds = self.subject_datasets[i]
                data, label = ds[local_idx]
                clinical = self.clinical_features[ds.subject_id]
                return data, label, clinical
        raise IndexError(f"Index {idx} out of range")


# ---------------------------------------------------------------------------
# Pretraining Trainer
# ---------------------------------------------------------------------------

class CACMLPretrainTrainer(BaseTrainer):
    """
    Pretrain CACML on all source subjects jointly.

    Trains the EEG Tower, Clinical Tower, and Classifier simultaneously using:
    - Cross-entropy loss on classification
    - Contrastive InfoNCE loss between EEG and clinical embeddings
    - L2 alignment loss for same-sample pairs

    Saves the best checkpoint based on validation accuracy.
    """

    def __init__(self, model=None, config=None, device=None, paths=None, logger=None, **kwargs):
        super().__init__(model=model, config=config, device=device, paths=paths, logger=logger)

        # Training hyperparameters
        self.epochs = self.config.get('training.epochs', 100)
        self.batch_size = self.config.get('training.batch_size', 64)
        self.lr = self.config.get('training.optimizer.lr', 1e-3)
        self.weight_decay = self.config.get('training.optimizer.weight_decay', 0.01)

        # CACML loss hyperparameters
        self.lambda_contrast = self.config.get('trainer.args.lambda_contrast', 1.0)
        self.lambda_align = self.config.get('trainer.args.lambda_align', 0.5)
        self.lambda_cls = self.config.get('trainer.args.lambda_cls', 1.0)
        self.tau = self.config.get('trainer.args.tau', 0.07)
        self.tau_clin = self.config.get('trainer.args.tau_clin', 0.07)
        self.contrast_mode = self.config.get('trainer.args.contrast_mode', 'clinical_anchor')
        self.use_hard_negative = self.config.get('trainer.args.use_hard_negative', True)
        self.k_hard = self.config.get('trainer.args.k_hard', 5)
        self.alpha_hard = self.config.get('trainer.args.alpha_hard', 2.0)
        self.val_ratio = self.config.get('trainer.args.val_ratio', 0.1)

        # CRITICAL: self-supervised pretraining mode
        # If True, pretraining uses ONLY contrastive + alignment losses (no labels).
        # This avoids data leakage in strict LOSO evaluation.
        # If False, classification loss is also used (faster convergence but
        # theoretically leaks label information into the pretrained representation).
        self.self_supervised_pretrain = self.config.get(
            'trainer.args.self_supervised_pretrain', True
        )

        # Scheme 3 (clinical_informed) and class_aware require MI labels for contrastive
        # positives. They cannot run in purely self-supervised mode.
        if self.self_supervised_pretrain and self.contrast_mode in ('class_aware', 'clinical_informed'):
            raise ValueError(
                f"contrast_mode='{self.contrast_mode}' requires labels. "
                "Set trainer.args.self_supervised_pretrain=false or use contrast_mode='clinical_anchor'."
            )

        self.criterion = CACMLLoss(
            lambda_contrast=self.lambda_contrast,
            lambda_align=self.lambda_align,
            lambda_cls=self.lambda_cls,
            tau=self.tau,
            tau_clin=self.tau_clin,
            contrast_mode=self.contrast_mode,
            use_hard_negative=self.use_hard_negative,
            k_hard=self.k_hard,
            alpha_hard=self.alpha_hard,
        )

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def train(self, datasets: Dict[str, Any]) -> Dict[str, Any]:
        subjects = datasets['subjects']
        dataset_info = datasets.get('info', {})
        dataset_cls = datasets.get('dataset_cls', XWStrokeDataset)

        self._log(f"Loading data for {len(subjects)} subjects for CACML pretraining...")
        subject_datasets = []
        for sid in subjects:
            try:
                ds = dataset_cls(subject_id=sid, dataset_info=dataset_info, mode='train')
                if len(ds) > 0:
                    subject_datasets.append(ds)
            except Exception as e:
                self._log(f"Warning: Failed to load subject {sid}: {e}", level='warning')

        if len(subject_datasets) == 0:
            raise ValueError("No valid subject datasets loaded for pretraining")

        self._log(f"Successfully loaded {len(subject_datasets)} subjects, "
                 f"total samples: {sum(len(ds) for ds in subject_datasets)}")

        pretrain_dataset = CACMLPretrainDataset(subject_datasets, dataset_info)

        # Train/validation split
        n_total = len(pretrain_dataset)
        n_val = max(1, int(n_total * self.val_ratio))
        n_train = n_total - n_val

        if n_val > 0:
            train_subset, val_subset = torch.utils.data.random_split(
                pretrain_dataset, [n_train, n_val],
                generator=torch.Generator().manual_seed(42)
            )
            val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)
        else:
            train_subset = pretrain_dataset
            val_loader = None

        train_loader = DataLoader(
            train_subset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

        best_val_acc = 0.0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(self.epochs):
            train_loss, train_acc, train_details = self._train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            log_msg = (f"Epoch {epoch+1:03d}/{self.epochs}: "
                      f"L={train_loss:.3f}(c={train_details['loss_contrast']:.3f}"
                      f" a={train_details['loss_align']:.3f}"
                      f" cls={train_details['loss_cls']:.3f})")

            if val_loader is not None:
                val_loss, val_acc, val_details = self._validate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self._save_best_checkpoint(epoch, best_val_acc)

                log_msg += (f" | Val L={val_loss:.3f}(c={val_details['loss_contrast']:.3f}"
                           f" a={val_details['loss_align']:.3f}"
                           f" cls={val_details['loss_cls']:.3f})")

            self._log(log_msg)

            # NOTE: torch.cuda.empty_cache() is DISABLED due to driver-level
            # segfaults after ~20-30 epochs on RTX 5090. PyTorch caches are
            # automatically reused; explicit empty_cache is unnecessary for
            # single-process training and triggers a known CUDA driver bug.
            # See: https://github.com/pytorch/pytorch/issues/124006
            pass

        # Save final checkpoint
        final_path = self.paths.checkpoints_dir / 'cacml_pretrained_final.pt'
        self.paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.save_checkpoint(str(final_path), best_val_acc=best_val_acc, history=history)
        self._log(f"Pretrained model saved to {final_path}")

        return {'history': history, 'best_val_acc': best_val_acc}

    def _train_epoch(self, dataloader: DataLoader) -> Tuple[float, float, Dict]:
        self.model.train()
        total_loss = 0.0
        total_loss_cls = 0.0
        total_loss_contrast = 0.0
        total_loss_align = 0.0
        correct = 0
        total = 0

        for data, labels, clinical in dataloader:
            data = data.to(self.device)
            labels = labels.to(self.device)
            clinical = clinical.to(self.device)

            self.optimizer.zero_grad()
            logits, f_eeg, f_clin = self.model(data, clinical, return_features=True)

            if self.self_supervised_pretrain:
                # Self-supervised: ignore classification loss, use contrastive + align only
                loss_contrast, loss_align = self._compute_contrastive_align_loss(
                    f_eeg, f_clin
                )
                loss = self.lambda_contrast * loss_contrast + self.lambda_align * loss_align
                loss_cls = torch.tensor(0.0, device=self.device)
            else:
                # Supervised: use full CACML loss (classification + contrastive + align)
                loss, loss_dict = self.criterion(logits, labels, f_eeg, f_clin)
                loss_cls = torch.tensor(loss_dict['loss_cls'], device=self.device)
                loss_contrast = torch.tensor(loss_dict['loss_contrast'], device=self.device)
                loss_align = torch.tensor(loss_dict['loss_align'], device=self.device)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * data.size(0)
            total_loss_cls += loss_cls.item() * data.size(0)
            total_loss_contrast += loss_contrast.item() * data.size(0)
            total_loss_align += loss_align.item() * data.size(0)
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        loss_details = {
            'loss': total_loss / total,
            'loss_cls': total_loss_cls / total,
            'loss_contrast': total_loss_contrast / total,
            'loss_align': total_loss_align / total,
        }
        return total_loss / total, correct / total, loss_details

    def _validate(self, dataloader: DataLoader) -> Tuple[float, float, Dict]:
        self.model.eval()
        total_loss = 0.0
        total_loss_cls = 0.0
        total_loss_contrast = 0.0
        total_loss_align = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, labels, clinical in dataloader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                clinical = clinical.to(self.device)

                logits, f_eeg, f_clin = self.model(data, clinical, return_features=True)

                if self.self_supervised_pretrain:
                    loss_contrast, loss_align = self._compute_contrastive_align_loss(
                        f_eeg, f_clin
                    )
                    loss = self.lambda_contrast * loss_contrast + self.lambda_align * loss_align
                    loss_cls = torch.tensor(0.0, device=self.device)
                else:
                    loss, loss_dict = self.criterion(logits, labels, f_eeg, f_clin)
                    loss_cls = torch.tensor(loss_dict['loss_cls'], device=self.device)
                    loss_contrast = torch.tensor(loss_dict['loss_contrast'], device=self.device)
                    loss_align = torch.tensor(loss_dict['loss_align'], device=self.device)

                total_loss += loss.item() * data.size(0)
                total_loss_cls += loss_cls.item() * data.size(0)
                total_loss_contrast += loss_contrast.item() * data.size(0)
                total_loss_align += loss_align.item() * data.size(0)
                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        loss_details = {
            'loss': total_loss / total,
            'loss_cls': total_loss_cls / total,
            'loss_contrast': total_loss_contrast / total,
            'loss_align': total_loss_align / total,
        }
        return total_loss / total, correct / total, loss_details

    def _compute_contrastive_align_loss(self, f_eeg, f_clin, labels=None):
        """
        Compute only contrastive + alignment losses (no classification).
        Used in self-supervised pretraining mode to avoid label leakage.

        NOTE: class_aware and clinical_informed modes require labels; if labels are
        not provided in self-supervised mode, this method falls back to the original
        clinical_anchor contrastive and emits a warning.
        """
        B = f_eeg.shape[0]
        device = f_eeg.device

        # Alignment loss
        loss_align = nn.functional.mse_loss(f_eeg, f_clin)

        # In self-supervised mode we typically don't have labels. The label-based
        # contrastive modes are not applicable here; enforce clinical_anchor.
        effective_mode = self.contrast_mode
        if effective_mode in ('class_aware', 'clinical_informed'):
            if labels is None:
                import warnings
                warnings.warn(
                    f"contrast_mode='{effective_mode}' requires labels; falling back to "
                    "clinical_anchor for self-supervised pretraining."
                )
                effective_mode = 'clinical_anchor'
            elif effective_mode == 'class_aware':
                loss_contrast = self.criterion._class_aware_contrastive(f_eeg, labels)
                return loss_contrast, loss_align
            elif effective_mode == 'clinical_informed':
                loss_contrast = self.criterion._clinical_informed_contrastive(f_eeg, f_clin, labels)
                return loss_contrast, loss_align

        # clinical_anchor (default self-supervised behavior)
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
            loss_e2c = nn.functional.cross_entropy(weighted_sim, labels_idx)
            loss_c2e = nn.functional.cross_entropy(weighted_sim.t(), labels_idx)
        else:
            labels_idx = torch.arange(B, device=device)
            loss_e2c = nn.functional.cross_entropy(sim, labels_idx)
            loss_c2e = nn.functional.cross_entropy(sim.t(), labels_idx)

        loss_contrast = (loss_e2c + loss_c2e) / 2.0
        return loss_contrast, loss_align

    def _save_best_checkpoint(self, epoch: int, val_acc: float):
        # Guard: only save if checkpoints are enabled AND we haven't saved
        # too recently (avoid I/O pressure and CUDA driver stress)
        save_enabled = self.config.get('training.save_checkpoints', True)
        if not save_enabled:
            return
        min_save_interval = self.config.get('trainer.args.min_save_interval', 5)
        if hasattr(self, '_last_save_epoch') and (epoch - self._last_save_epoch) < min_save_interval:
            return
        self._last_save_epoch = epoch

        checkpoint_path = self.paths.checkpoints_dir / 'cacml_pretrained_best.pt'
        self.paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        # Save WITHOUT optimizer state_dict to reduce size and I/O pressure
        self.save_checkpoint(
            str(checkpoint_path),
            epoch=epoch,
            val_acc=val_acc,
        )


# ---------------------------------------------------------------------------
# LOSO Evaluation Trainer
# ---------------------------------------------------------------------------

class CACMLLOSOSTrainer(BaseTrainer):
    """
    Leave-One-Subject-Out evaluation for CACML with n-shot fine-tuning.

    For each subject (held out as target):
    1. Load pretrained CACML weights
    2. (Optional) Sample n_shot trials as support set
    3. (Optional) Fine-tune classifier on support set with frozen backbone
    4. Evaluate on remaining test trials

    Config args:
        pretrained_path: Path to pretrained CACML checkpoint
        n_shot: Number of support trials per subject (0 for zero-shot)
        finetune_epochs: Epochs for classifier fine-tuning
        lr_finetune: Learning rate for fine-tuning
        freeze_backbone: Whether to freeze EEG and Clinical towers during fine-tuning
    """

    def __init__(self, model=None, config=None, device=None, paths=None, logger=None, **kwargs):
        super().__init__(model=model, config=config, device=device, paths=paths, logger=logger)

        self.pretrained_path = self.config.get('trainer.args.pretrained_path')
        self.n_shot = self.config.get('trainer.args.n_shot', 20)
        self.finetune_epochs = self.config.get('trainer.args.finetune_epochs', 50)
        self.lr_finetune = float(self.config.get('trainer.args.lr_finetune', 1e-4))
        self.freeze_backbone = self.config.get('trainer.args.freeze_backbone', True)
        self.seed = self.config.get('experiment.seed', 42)

        # Reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.criterion = nn.CrossEntropyLoss()

    def train(self, datasets: Dict[str, Any]) -> Dict[str, Any]:
        subjects = datasets['subjects']
        dataset_info = datasets.get('info', {})
        dataset_cls = datasets.get('dataset_cls', XWStrokeDataset)

        all_results = []

        for target_subject in subjects:
            self._log(f"\n{'='*60}")
            self._log(f"LOSO Target Subject: {target_subject:02d}")
            self._log(f"{'='*60}")

            # Load target subject data
            try:
                target_dataset = dataset_cls(
                    subject_id=target_subject, dataset_info=dataset_info, mode='train'
                )
            except Exception as e:
                self._log(f"Error loading subject {target_subject}: {e}", level='warning')
                continue

            if len(target_dataset) == 0:
                self._log(f"Subject {target_subject} has no data, skipping", level='warning')
                continue

            # Build fresh model with correct dimensions
            model = self._build_model(target_dataset)

            # Load pretrained weights
            if self.pretrained_path and Path(self.pretrained_path).exists():
                self._load_pretrained_weights(model)
            else:
                self._log("WARNING: No valid pretrained_path. Training from scratch.", level='warning')

            # Support / test split
            if self.n_shot > 0:
                support_indices = self._sample_n_shot(target_dataset, self.n_shot)
                test_indices = [i for i in range(len(target_dataset)) if i not in support_indices]
            else:
                support_indices = []
                test_indices = list(range(len(target_dataset)))

            if len(test_indices) == 0:
                self._log(f"No test data for subject {target_subject}, skipping", level='warning')
                continue

            # Fine-tune on support set
            if self.n_shot > 0 and len(support_indices) > 0:
                self._finetune_classifier(model, target_dataset, support_indices)

            # Evaluate on test set
            test_acc = self._evaluate(model, target_dataset, test_indices)

            all_results.append({
                'subject_id': target_subject,
                'accuracy': test_acc,
                'n_support': len(support_indices),
                'n_test': len(test_indices),
            })
            self._log(f"Subject {target_subject:02d}: Accuracy = {test_acc:.4f} "
                     f"(support={len(support_indices)}, test={len(test_indices)})")

        # Aggregate statistics
        accs = [r['accuracy'] for r in all_results]
        mean_acc = float(np.mean(accs)) if accs else 0.0
        std_acc = float(np.std(accs)) if accs else 0.0

        self._log(f"\n{'='*60}")
        self._log(f"CACML LOSO Summary ({self.n_shot}-shot)")
        self._log(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        self._log(f"Subjects evaluated: {len(all_results)}/{len(subjects)}")
        self._log(f"{'='*60}")

        return {
            'subjects': all_results,
            'mean': mean_acc,
            'std': std_acc,
            'n_shot': self.n_shot,
        }

    def _build_model(self, dataset) -> nn.Module:
        """Build a fresh CACML model with dimensions inferred from dataset."""
        model_config = {
            'type': self.config.get('model.type', 'CACML'),
            'args': (self.config.get('model.args') or {}).copy()
        }

        # Auto-populate dimensions from dataset
        if 'num_channels' not in model_config['args']:
            model_config['args']['num_channels'] = dataset.data.shape[2]
        if 'num_classes' not in model_config['args']:
            model_config['args']['num_classes'] = len(set(dataset.labels))
        if 'num_bands' not in model_config['args']:
            model_config['args']['num_bands'] = dataset.data.shape[1]
        if 'input_length' not in model_config['args']:
            model_config['args']['input_length'] = dataset.data.shape[3]

        model = build_from_config(model_config, MODELS)
        return model.to(self.device)

    def _load_pretrained_weights(self, model: nn.Module):
        """Load pretrained weights, skipping incompatible layers."""
        self._log(f"Loading pretrained weights from: {self.pretrained_path}")
        checkpoint = torch.load(self.pretrained_path, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        model_dict = model.state_dict()
        compatible = {}
        skipped = []
        for k, v in state_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                compatible[k] = v
            else:
                skipped.append(k)

        model_dict.update(compatible)
        model.load_state_dict(model_dict, strict=False)
        self._log(f"Loaded {len(compatible)}/{len(state_dict)} layers "
                 f"({len(skipped)} skipped)")

    def _sample_n_shot(self, dataset, n_trials: int) -> List[int]:
        """
        Stratified sampling at the ORIGINAL TRIAL level (not window level).

        Each subject has 40 original trials. Sliding window creates ~5 windows
        per trial (200 total). To avoid information leakage, we sample trials
        first, then take ALL windows belonging to those trials.

        Args:
            dataset: Dataset with .labels and .group_ids attributes
            n_trials: Number of original trials to sample (not windows)

        Returns:
            List of window indices belonging to the selected trials
        """
        group_ids = dataset.group_ids
        labels = dataset.labels

        # Unique original trials and their labels
        unique_groups = np.unique(group_ids)
        group_to_label = {}
        for g in unique_groups:
            mask = group_ids == g
            group_to_label[g] = labels[mask][0]

        # Stratified sampling by class label
        unique_labels = np.unique(list(group_to_label.values()))
        n_per_class = max(1, n_trials // len(unique_labels))

        selected_groups = []
        for label in unique_labels:
            groups_with_label = [g for g, l in group_to_label.items() if l == label]
            if len(groups_with_label) >= n_per_class:
                chosen = np.random.choice(groups_with_label, n_per_class, replace=False)
                selected_groups.extend(chosen.tolist())
            else:
                selected_groups.extend(groups_with_label)

        # Fill remaining from unused trials
        remaining = [g for g in unique_groups if g not in selected_groups]
        while len(selected_groups) < n_trials and remaining:
            idx = int(np.random.choice(remaining))
            selected_groups.append(idx)
            remaining.remove(idx)

        selected_groups = selected_groups[:n_trials]

        # Convert trial (group) IDs to all their window indices
        selected_indices = []
        for g in selected_groups:
            selected_indices.extend(np.where(group_ids == g)[0].tolist())

        return selected_indices

    def _finetune_classifier(self, model: nn.Module, dataset, support_indices: List[int]):
        """Fine-tune the classifier on support set with frozen backbone."""
        support_dataset = Subset(dataset, support_indices)
        support_loader = DataLoader(
            support_dataset,
            batch_size=min(len(support_indices), 32),
            shuffle=True,
        )

        # Freeze backbone towers
        if self.freeze_backbone:
            for name, param in model.named_parameters():
                if any(kw in name for kw in ['eeg_backbone', 'eeg_proj', 'clinical_tower']):
                    param.requires_grad = False

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        self._log(f"Fine-tuning: {trainable:,}/{total:,} parameters trainable")

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr_finetune,
        )

        model.train()
        for epoch in range(self.finetune_epochs):
            epoch_loss = 0.0
            for data, labels in support_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                logits = model(data)  # No clinical features needed for fine-tuning
                loss = self.criterion(logits, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

        self._log(f"Fine-tuned on {len(support_indices)} samples for {self.finetune_epochs} epochs")

    def _evaluate(self, model: nn.Module, dataset, test_indices: List[int]) -> float:
        """Evaluate model on test indices."""
        test_dataset = Subset(dataset, test_indices)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                logits = model(data)
                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        return correct / total if total > 0 else 0.0
