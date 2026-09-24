"""
Generic n-shot Leave-One-Subject-Out (LOSO) evaluation trainer.

Loads a pretrained model, optionally samples n original trials from each
held-out subject as a support set, fine-tunes (optionally with frozen backbone),
and evaluates on the remaining test windows.

Modeled after CACMLLOSOSTrainer but model-agnostic.
"""

import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from core.base.base_trainer import BaseTrainer
from core.registry import MODELS, build_from_config
from data.datasets import XWStrokeDataset


class NShotLOSOTrainer(BaseTrainer):
    """
    N-shot LOSO evaluation for any EEG classifier.

    Config args:
        pretrained_path: Path to pretrained checkpoint
        n_shot: Number of support trials per subject (0 for zero-shot)
        finetune_epochs: Epochs for fine-tuning
        lr_finetune: Learning rate for fine-tuning
        freeze_backbone: Whether to freeze all modules except 'classifier'
        batch_size_finetune: Batch size for support-set fine-tuning
    """

    def __init__(self, model=None, config=None, device=None, paths=None, logger=None, **kwargs):
        super().__init__(model=model, config=config, device=device, paths=paths, logger=logger)

        self.pretrained_path = self.config.get('trainer.args.pretrained_path')
        self.n_shot = self.config.get('trainer.args.n_shot', 20)
        self.finetune_epochs = self.config.get('trainer.args.finetune_epochs', 50)
        self.lr_finetune = float(self.config.get('trainer.args.lr_finetune', 1e-4))
        self.freeze_backbone = self.config.get('trainer.args.freeze_backbone', True)
        self.batch_size_finetune = self.config.get('trainer.args.batch_size_finetune', 32)
        self.seed = self.config.get('experiment.seed', 42)

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

            model = self._build_model(target_dataset)

            if self.pretrained_path and Path(self.pretrained_path).exists():
                self._load_pretrained_weights(model)
            else:
                self._log("WARNING: No valid pretrained_path. Training from scratch.", level='warning')

            if self.n_shot > 0:
                support_indices = self._sample_n_shot(target_dataset, self.n_shot)
                test_indices = [i for i in range(len(target_dataset)) if i not in support_indices]
            else:
                support_indices = []
                test_indices = list(range(len(target_dataset)))

            if len(test_indices) == 0:
                self._log(f"No test data for subject {target_subject}, skipping", level='warning')
                continue

            if self.n_shot > 0 and len(support_indices) > 0:
                self._finetune(model, target_dataset, support_indices)

            test_acc = self._evaluate(model, target_dataset, test_indices)

            all_results.append({
                'subject_id': target_subject,
                'accuracy': test_acc,
                'n_support': len(support_indices),
                'n_test': len(test_indices),
            })
            self._log(f"Subject {target_subject:02d}: Accuracy = {test_acc:.4f} "
                      f"(support={len(support_indices)}, test={len(test_indices)})")

        accs = [r['accuracy'] for r in all_results]
        mean_acc = float(np.mean(accs)) if accs else 0.0
        std_acc = float(np.std(accs)) if accs else 0.0

        self._log(f"\n{'='*60}")
        self._log(f"N-shot LOSO Summary ({self.n_shot}-shot)")
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
        model_config = {
            'type': self.config.get('model.type'),
            'args': (self.config.get('model.args') or {}).copy()
        }

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
        self._log(f"Loaded {len(compatible)}/{len(state_dict)} layers ({len(skipped)} skipped)")

    def _sample_n_shot(self, dataset, n_trials: int) -> List[int]:
        """
        Stratified sampling at the original trial level (not window level).
        """
        group_ids = dataset.group_ids
        labels = dataset.labels

        unique_groups = np.unique(group_ids)
        group_to_label = {g: labels[group_ids == g][0] for g in unique_groups}
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

        remaining = [g for g in unique_groups if g not in selected_groups]
        while len(selected_groups) < n_trials and remaining:
            idx = int(np.random.choice(remaining))
            selected_groups.append(idx)
            remaining.remove(idx)

        selected_groups = selected_groups[:n_trials]

        selected_indices = []
        for g in selected_groups:
            selected_indices.extend(np.where(group_ids == g)[0].tolist())

        return selected_indices

    def _finetune(self, model: nn.Module, dataset, support_indices: List[int]):
        support_dataset = Subset(dataset, support_indices)
        support_loader = DataLoader(
            support_dataset,
            batch_size=min(len(support_indices), self.batch_size_finetune),
            shuffle=True,
            num_workers=0,
        )

        if self.freeze_backbone:
            # If the model exposes a 'classifier' module, freeze everything else.
            # Otherwise, freeze all parameters (less common).
            if hasattr(model, 'classifier'):
                for param in model.parameters():
                    param.requires_grad = False
                for param in model.classifier.parameters():
                    param.requires_grad = True
            else:
                self._log("Model has no 'classifier' attribute; fine-tuning all parameters", level='warning')

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
                logits = model(data)
                loss = self.criterion(logits, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

        self._log(f"Fine-tuned on {len(support_indices)} samples for {self.finetune_epochs} epochs")

    def _evaluate(self, model: nn.Module, dataset, test_indices: List[int]) -> float:
        test_dataset = Subset(dataset, test_indices)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

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
