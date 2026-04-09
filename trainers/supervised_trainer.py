"""
Supervised learning trainer with k-fold cross-validation.
"""

import math
import copy
from typing import Any, Dict, List
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
import numpy as np

from core.base.base_trainer import BaseTrainer
from utils.visualization import plot_training_history, plot_subject_comparison


class SupervisedTrainer(BaseTrainer):
    """
    Trainer for supervised learning with k-fold cross-validation.
    
    Supports subject-wise training with group k-fold cross-validation
to prevent data leakage from overlapping windows.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the supervised trainer.
        
        Args:
            **kwargs: Passed to BaseTrainer
        """
        super().__init__(**kwargs)
        
        # Training configuration
        self.epochs = self.config.get('training.epochs', 100)
        self.batch_size = self.config.get('training.batch_size', 32)
        self.learning_rate = self.config.get('training.optimizer.lr', 1e-3)
        self.weight_decay = self.config.get('training.optimizer.weight_decay', 0.01)
        self.k_folds = self.config.get('training.k_folds', 5)
        
        # Store model class and args for reinitialization
        self.model_type = self.config.get('model.type')
        self.model_args = self.config.get('model.args', {})
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Checkpoint configuration
        self.save_checkpoints = self.config.get('training.save_checkpoints', True)
        self.save_optimizer_state = self.config.get('training.save_optimizer_state', False)
        self._checkpoint_warned = False  # Track if we've already warned about save failures
        
    def train(self, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run training for all subjects with lazy data loading.
        
        Each subject's data is loaded on-demand to minimize memory usage.
        
        Args:
            datasets: Dictionary containing 'subjects', 'info', and 'dataset_cls'
                     (for lazy loading) or 'instances' (for pre-loaded data)
            
        Returns:
            Dictionary containing training results
        """
        subjects = datasets['subjects']
        dataset_info = datasets.get('info', {})
        
        # Support both lazy loading and pre-loaded instances
        use_lazy_loading = 'dataset_cls' in datasets
        dataset_cls = datasets.get('dataset_cls')
        subject_instances = datasets.get('instances', {})
        
        all_results = []
        all_histories = []
        
        self._log("=" * 60)
        self._log("Starting Supervised Training")
        self._log(f"Subjects: {subjects}")
        self._log(f"Epochs: {self.epochs}, K-folds: {self.k_folds}")
        self._log("=" * 60)
        
        for subject_id in subjects:
            self._log(f"\nProcessing Subject: {subject_id}")
            self._log("-" * 40)
            
            # Load data on-demand if using lazy loading
            if use_lazy_loading:
                dataset = dataset_cls(subject_id=subject_id, dataset_info=dataset_info)
            else:
                dataset = subject_instances[subject_id]
            
            # Train this subject
            subject_results, subject_history = self._train_subject(subject_id, dataset)
            
            all_results.append(subject_results)
            all_histories.append({
                'subject_id': subject_id,
                'history': subject_history
            })
            
            # Plot individual subject history
            subject_viz_dir = self.paths.get_subject_viz_dir(subject_id)
            plot_training_history(subject_history, subject_viz_dir, 
                                 title=f'Subject {subject_id} Training History')
            
            # Free memory after training this subject
            if use_lazy_loading:
                del dataset
                import gc
                gc.collect()
        
        # Plot comparison across subjects
        plot_subject_comparison(all_results, self.paths.viz_dir)
        
        # Calculate overall statistics
        final_results = {
            'subjects': all_results,
            'overall_mean': np.mean([r['avg_val_acc'] for r in all_results]),
            'overall_std': np.std([r['avg_val_acc'] for r in all_results]),
        }
        
        self._log("\n" + "=" * 60)
        self._log("Training Complete")
        self._log(f"Overall: {final_results['overall_mean']:.4f} ± {final_results['overall_std']:.4f}")
        self._log("=" * 60)
        
        return final_results
    
    def _train_subject(self, subject_id: int, dataset) -> tuple:
        """
        Train a single subject with k-fold cross-validation.
        
        Args:
            subject_id: Subject identifier
            dataset: Dataset instance for this subject
            
        Returns:
            Tuple of (results_dict, training_history)
        """
        # Setup k-fold
        kfold = GroupKFold(n_splits=self.k_folds)
        
        fold_histories = []
        fold_best_accuracies = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(
            kfold.split(dataset, groups=dataset.group_ids)
        ):
            self._log(f"  Fold {fold_idx + 1}/{self.k_folds}")
            
            # Create data loaders
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            
            train_loader = DataLoader(
                train_subset, 
                batch_size=self.batch_size,
                shuffle=True,
                generator=torch.Generator(device=self.device)
            )
            val_loader = DataLoader(
                val_subset,
                batch_size=self.batch_size,
                shuffle=False,
                generator=torch.Generator(device=self.device)
            )
            
            # Create fresh model for this fold
            fold_model = self._create_model_for_fold(dataset)
            optimizer = torch.optim.Adam(
                fold_model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
            
            # Training loop
            fold_history = {
                'train_loss': [],
                'val_loss': [],
                'train_acc': [],
                'val_acc': []
            }
            best_val_acc = 0.0
            
            # Single progress bar for epochs with real-time metrics
            epoch_progress_bar = tqdm(
                range(self.epochs),
                desc=f'Sub{subject_id} Fold{fold_idx+1}',
                bar_format='{desc} |{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}'
            )
            
            for epoch in epoch_progress_bar:
                self.current_epoch = epoch
                
                # Cosine annealing learning rate
                lr = self._compute_lr(epoch)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                
                # Train and validate
                train_loss, train_acc = self._train_epoch(fold_model, train_loader, optimizer)
                val_loss, val_acc = self._validate(fold_model, val_loader)
                
                # Record history
                fold_history['train_loss'].append(train_loss)
                fold_history['train_acc'].append(train_acc)
                fold_history['val_loss'].append(val_loss)
                fold_history['val_acc'].append(val_acc)
                
                # Update progress bar with all metrics in compact format
                epoch_progress_bar.set_postfix_str(
                    f"LR:{lr:.1e} T:{train_loss:.3f}/{train_acc:.3f} V:{val_loss:.3f}/{val_acc:.3f}"
                )
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self._save_fold_checkpoint(fold_model, optimizer, subject_id, 
                                              fold_idx, epoch, val_acc)
                
                self.global_step += 1
            
            epoch_progress_bar.close()
            fold_histories.append(fold_history)
            fold_best_accuracies.append(best_val_acc)
            self._log(f"    Best Val Acc: {best_val_acc:.4f}")
        
        # Calculate subject statistics
        subject_avg = np.mean(fold_best_accuracies)
        subject_std = np.std(fold_best_accuracies)
        
        results = {
            'subject_id': subject_id,
            'avg_val_acc': subject_avg,
            'std_val_acc': subject_std,
            'fold_accuracies': fold_best_accuracies
        }
        
        return results, fold_histories
    
    def _create_model_for_fold(self, dataset) -> nn.Module:
        """
        Create a fresh model instance for a new fold.
        
        Args:
            dataset: Dataset instance (for inferring model parameters)
            
        Returns:
            New model instance
        """
        from core.registry import MODELS, build_from_config
        
        # Build model config
        model_config = {
            'type': self.model_type,
            'args': self.model_args.copy()
        }
        
        # Auto-populate model arguments from dataset info
        if 'num_channels' not in model_config['args']:
            # Get from data shape: (N, bands, channels, time)
            model_config['args']['num_channels'] = dataset.data.shape[2]
        if 'num_classes' not in model_config['args']:
            model_config['args']['num_classes'] = len(set(dataset.labels))
        if 'num_bands' not in model_config['args']:
            model_config['args']['num_bands'] = dataset.data.shape[1]
        if 'input_length' not in model_config['args']:
            model_config['args']['input_length'] = dataset.data.shape[3]
        
        # Build model
        model = build_from_config(model_config, MODELS)
        return model.to(self.device)
    
    def _compute_lr(self, epoch: int) -> float:
        """
        Compute learning rate with cosine annealing.
        
        Args:
            epoch: Current epoch
            
        Returns:
            Learning rate for this epoch
        """
        return (1 + math.cos(epoch * math.pi / self.epochs)) * self.learning_rate / 2
    
    def _train_epoch(self, model: nn.Module, dataloader: DataLoader, 
                    optimizer: torch.optim.Optimizer) -> tuple:
        """
        Train for one epoch.
        
        Args:
            model: Model to train
            dataloader: Training data loader
            optimizer: Optimizer
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _validate(self, model: nn.Module, dataloader: DataLoader) -> tuple:
        """
        Validate the model.
        
        Args:
            model: Model to validate
            dataloader: Validation data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _save_fold_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                             subject_id: int, fold: int, epoch: int, 
                             metric: float) -> None:
        """
        Save checkpoint for a specific fold.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            subject_id: Subject identifier
            fold: Fold number
            epoch: Current epoch
            metric: Performance metric
        """
        if not self.save_checkpoints:
            return
        
        # 构建 checkpoint，根据配置决定是否包含优化器状态
        checkpoint = {
            'epoch': epoch,
            'subject_id': subject_id,
            'fold': fold,
            'model_state_dict': model.state_dict(),
            'metric': metric
        }
        
        # 可选：保存优化器状态（会显著增加文件大小）
        if self.save_optimizer_state:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        checkpoint_path = self.paths.get_checkpoint_path(
            subject_id=subject_id,
            fold=fold,
            metric=metric
        )
        
        try:
            torch.save(checkpoint, checkpoint_path)
        except (RuntimeError, IOError, OSError) as e:
            # 保存失败时只在第一次记录警告，避免频繁打断进度条
            if not self._checkpoint_warned:
                self._log(f"    Warning: Checkpoint save failed (will retry silently): {str(e)[:80]}...", level='warning')
                self._checkpoint_warned = True
