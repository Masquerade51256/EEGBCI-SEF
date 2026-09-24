"""
Simple supervised pretraining trainer for generic EEG models.

Trains a single model on all source subjects jointly and saves the best
validation checkpoint. Does not require clinical features or contrastive loss.
"""

import math
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from core.base.base_trainer import BaseTrainer
from core.registry import MODELS, build_from_config


class SimplePretrainTrainer(BaseTrainer):
    """
    Supervised pretraining on multiple source subjects.

    Config args:
        val_ratio: Fraction of combined data to hold out for validation (default 0.1)
        min_save_interval: Minimum epochs between checkpoint saves (default 5)
    """

    def __init__(self, model=None, config=None, device=None, paths=None, logger=None, **kwargs):
        super().__init__(model=model, config=config, device=device, paths=paths, logger=logger)

        self.epochs = self.config.get('training.epochs', 100)
        self.batch_size = self.config.get('training.batch_size', 64)
        self.lr = self.config.get('training.optimizer.lr', 1e-3)
        self.weight_decay = self.config.get('training.optimizer.weight_decay', 0.01)

        self.val_ratio = self.config.get('trainer.args.val_ratio', 0.1)
        self.min_save_interval = self.config.get('trainer.args.min_save_interval', 5)

        self.seed = self.config.get('experiment.seed', 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.criterion = nn.CrossEntropyLoss()
        self.best_val_acc = 0.0
        self._last_save_epoch = -self.min_save_interval

    def train(self, datasets: Dict[str, Any]) -> Dict[str, Any]:
        subjects = datasets['subjects']
        dataset_info = datasets.get('info', {})
        dataset_cls = datasets.get('dataset_cls')

        if not dataset_cls:
            raise ValueError("SimplePretrainTrainer requires 'dataset_cls'")

        # Load all source subject datasets
        subject_datasets = []
        for sid in subjects:
            try:
                ds = dataset_cls(subject_id=sid, dataset_info=dataset_info, mode='train')
                if len(ds) > 0:
                    subject_datasets.append(ds)
            except Exception as e:
                self._log(f"Error loading subject {sid}: {e}", level='warning')

        if not subject_datasets:
            raise ValueError("No valid subject datasets found for pretraining")

        full_dataset = ConcatDataset(subject_datasets)
        self._log(f"Combined dataset: {len(full_dataset)} samples from {len(subject_datasets)} subjects")

        # Train/val split
        n_total = len(full_dataset)
        n_val = int(n_total * self.val_ratio)
        n_train = n_total - n_val
        indices = list(range(n_total))
        random.shuffle(indices)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        train_subset = torch.utils.data.Subset(full_dataset, train_indices)
        val_subset = torch.utils.data.Subset(full_dataset, val_indices)

        train_loader = DataLoader(
            train_subset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_subset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        # Build model
        model = self._build_model(subject_datasets[0])
        model.to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self._log(f"Model: {total_params:,} params ({trainable_params:,} trainable)")

        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(self.epochs):
            self.current_epoch = epoch
            lr = self._compute_lr(epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            train_loss, train_acc = self._train_epoch(model, train_loader, optimizer)
            val_loss, val_acc = self._validate(model, val_loader)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            self._log(f"Epoch {epoch+1}/{self.epochs}: "
                      f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                      f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_best_checkpoint(model, epoch, val_acc)

            self.global_step += 1

        # Save final checkpoint
        self._save_checkpoint(model, epoch=-1, val_acc=val_acc, filename='pretrained_final.pt')

        return {
            'history': history,
            'best_val_acc': self.best_val_acc,
            'n_train': n_train,
            'n_val': n_val,
        }

    def _build_model(self, sample_dataset) -> nn.Module:
        model_config = {
            'type': self.config.get('model.type'),
            'args': (self.config.get('model.args') or {}).copy()
        }

        if 'num_channels' not in model_config['args']:
            model_config['args']['num_channels'] = sample_dataset.data.shape[2]
        if 'num_classes' not in model_config['args']:
            model_config['args']['num_classes'] = len(set(sample_dataset.labels))
        if 'num_bands' not in model_config['args']:
            model_config['args']['num_bands'] = sample_dataset.data.shape[1]
        if 'input_length' not in model_config['args']:
            model_config['args']['input_length'] = sample_dataset.data.shape[3]

        model = build_from_config(model_config, MODELS)
        return model.to(self.device)

    def _compute_lr(self, epoch: int) -> float:
        return (1 + math.cos(epoch * math.pi / self.epochs)) * self.lr / 2

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

    def _save_best_checkpoint(self, model: nn.Module, epoch: int, val_acc: float):
        save_enabled = self.config.get('training.save_checkpoints', True)
        if not save_enabled:
            return
        if epoch - self._last_save_epoch < self.min_save_interval and epoch > 0:
            return
        self._last_save_epoch = epoch
        self._save_checkpoint(model, epoch, val_acc, filename='pretrained_best.pt')

    def _save_checkpoint(self, model: nn.Module, epoch: int, val_acc: float, filename: str):
        self.paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        path = self.paths.checkpoints_dir / filename
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_acc': val_acc,
        }, path)
        self._log(f"Saved checkpoint: {path}")
