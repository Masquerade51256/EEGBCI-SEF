"""
Within-subject k-fold cross-validation trainer.

For each subject, runs k-fold CV at the original-trial level to avoid
information leakage from sliding-window overlap. Reports per-subject and
overall mean accuracy.
"""

import random
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

from core.base.base_trainer import BaseTrainer
from core.registry import MODELS, build_from_config
from data.datasets import XWStrokeDataset, get_clinical_feature_vector


class WithinSubjectKFoldTrainer(BaseTrainer):
    """
    Within-subject k-fold CV for EEG classifiers.

    Config args:
        n_folds: Number of folds (default 5)
        epochs: Training epochs per fold
        batch_size: Batch size
        lr: Learning rate
        weight_decay: Weight decay
        early_stop_patience: Early stopping patience (default 10)
    """

    def __init__(self, model=None, config=None, device=None, paths=None, logger=None, **kwargs):
        super().__init__(model=model, config=config, device=device, paths=paths, logger=logger)

        self.n_folds = self.config.get('trainer.args.n_folds', 5)
        self.epochs = self.config.get('trainer.args.epochs',
                                       self.config.get('training.epochs', 100))
        self.batch_size = self.config.get('trainer.args.batch_size',
                                          self.config.get('training.batch_size', 32))
        self.lr = self.config.get('trainer.args.lr',
                                  self.config.get('training.optimizer.lr', 1e-3))
        self.weight_decay = self.config.get('trainer.args.weight_decay',
                                            self.config.get('training.optimizer.weight_decay', 0.01))
        self.early_stop_patience = self.config.get('trainer.args.early_stop_patience', 10)
        self.seed = self.config.get('experiment.seed', 42)

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.criterion = nn.CrossEntropyLoss()

    def train(self, datasets: Dict[str, Any]) -> Dict[str, Any]:
        subjects = datasets['subjects']
        dataset_info = datasets.get('info', {})
        dataset_cls = datasets.get('dataset_cls', XWStrokeDataset)

        all_subject_results = []

        for subject_id in subjects:
            self._log(f"\n{'='*60}")
            self._log(f"Subject {subject_id:02d}")
            self._log(f"{'='*60}")

            try:
                dataset = dataset_cls(subject_id=subject_id, dataset_info=dataset_info, mode='train')
            except Exception as e:
                self._log(f"Error loading subject {subject_id}: {e}", level='warning')
                continue

            if len(dataset) == 0:
                self._log(f"Subject {subject_id} has no data, skipping", level='warning')
                continue

            fold_accs = self._run_subject_kfold(subject_id, dataset, dataset_info)
            mean_acc = float(np.mean(fold_accs)) if fold_accs else 0.0
            std_acc = float(np.std(fold_accs)) if fold_accs else 0.0

            all_subject_results.append({
                'subject_id': subject_id,
                'fold_accs': fold_accs,
                'mean': mean_acc,
                'std': std_acc,
            })

            self._log(f"Subject {subject_id:02d}: {mean_acc*100:.2f}% ± {std_acc*100:.2f}% "
                      f"(folds: {[f'{a*100:.1f}%' for a in fold_accs]})")

        overall_accs = [r['mean'] for r in all_subject_results]
        overall_mean = float(np.mean(overall_accs)) if overall_accs else 0.0
        overall_std = float(np.std(overall_accs)) if overall_accs else 0.0

        self._log(f"\n{'='*60}")
        self._log(f"Overall Within-Subject {self.n_folds}-fold CV: "
                  f"{overall_mean*100:.2f}% ± {overall_std*100:.2f}% "
                  f"(n={len(all_subject_results)} subjects)")
        self._log(f"{'='*60}")

        return {
            'subjects': all_subject_results,
            'overall_mean': overall_mean,
            'overall_std': overall_std,
            'n_folds': self.n_folds,
        }

    def _run_subject_kfold(self, subject_id: int, dataset, dataset_info: Dict) -> List[float]:
        group_ids = dataset.group_ids
        labels = dataset.labels
        unique_groups = np.unique(group_ids)
        group_to_label = {g: labels[group_ids == g][0] for g in unique_groups}

        # Split original trials (groups) into folds
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed + subject_id)
        fold_accs = []

        for fold_idx, (train_groups_idx, test_groups_idx) in enumerate(kf.split(unique_groups)):
            train_groups = unique_groups[train_groups_idx]
            test_groups = unique_groups[test_groups_idx]

            train_indices = [i for i, g in enumerate(group_ids) if g in train_groups]
            test_indices = [i for i, g in enumerate(group_ids) if g in test_groups]

            model = self._build_model(dataset)
            model.to(self.device)

            optimizer = torch.optim.Adam(
                model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )

            train_loader = DataLoader(
                Subset(dataset, train_indices),
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
            )
            val_loader = DataLoader(
                Subset(dataset, test_indices),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
            )

            best_val_acc = 0.0
            patience_counter = 0

            for epoch in range(self.epochs):
                train_loss, train_acc = self._train_epoch(model, train_loader, optimizer)
                val_loss, val_acc = self._validate(model, val_loader)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.early_stop_patience:
                    self._log(f"  Fold {fold_idx+1}: early stop at epoch {epoch+1}, "
                              f"best val acc {best_val_acc*100:.2f}%")
                    break

            fold_accs.append(best_val_acc)
            self._log(f"  Fold {fold_idx+1}/{self.n_folds}: {best_val_acc*100:.2f}%")

        return fold_accs

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

    def _train_epoch(self, model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for data, labels in loader:
            data = data.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            logits = model(data)
            loss = self.criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        return total_loss / total if total > 0 else 0.0, correct / total if total > 0 else 0.0

    def _validate(self, model: nn.Module, loader: DataLoader):
        model.eval()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for data, labels in loader:
                data = data.to(self.device)
                labels = labels.to(self.device)

                logits = model(data)
                loss = self.criterion(logits, labels)

                total_loss += loss.item() * data.size(0)
                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        return total_loss / total if total > 0 else 0.0, correct / total if total > 0 else 0.0
