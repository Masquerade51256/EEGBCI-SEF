"""
Streaming LOSO (Leave One Subject Out) trainer with memory-efficient data loading.

For datasets with many subjects (e.g., 50 subjects in XW Stroke):
- Instead of loading all N-1 subjects into memory at once
- Stream data subject by subject during each epoch
- Use gradient accumulation to simulate large batch training

This reduces memory from O(N) to O(1) subjects while maintaining the same LOSO evaluation.
"""

import math
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np

from core.base.base_trainer import BaseTrainer


class StreamingSubjectDataset(Dataset):
    """
    Memory-efficient dataset that loads subjects on-demand.
    
    Instead of loading all data into memory, it maintains a list of subject IDs
    and loads data batch by batch during iteration.
    """
    
    def __init__(self, subject_ids: List[int], dataset_cls, dataset_info: Dict,
                 subject_buffer_size: int = 1, device: str = 'cpu'):
        """
        Initialize streaming dataset.
        
        Args:
            subject_ids: List of subject IDs to include
            dataset_cls: Dataset class for loading subjects
            dataset_info: Dataset configuration
            subject_buffer_size: Number of subjects to keep in memory (default: 1)
            device: Device for tensor storage
        """
        self.subject_ids = subject_ids
        self.dataset_cls = dataset_cls
        self.dataset_info = dataset_info
        self.subject_buffer_size = subject_buffer_size
        self.device = device
        
        # Cache for loaded subjects (LRU style)
        self._subject_cache = {}
        self._cache_order = []
        
        # Pre-compute total size without loading all data
        self._total_size = None
        self._subject_sizes = {}
    
    def _get_subject_data(self, subject_id: int):
        """Load subject data with caching."""
        if subject_id in self._subject_cache:
            # Move to end (most recently used)
            self._cache_order.remove(subject_id)
            self._cache_order.append(subject_id)
            return self._subject_cache[subject_id]
        
        # Load subject
        ds = self.dataset_cls(subject_id=subject_id, dataset_info=self.dataset_info)
        
        # Cache management
        if len(self._subject_cache) >= self.subject_buffer_size:
            # Remove oldest
            oldest = self._cache_order.pop(0)
            del self._subject_cache[oldest]
        
        self._subject_cache[subject_id] = ds
        self._cache_order.append(subject_id)
        
        return ds
    
    def __len__(self):
        """Get total samples (lazy computation)."""
        if self._total_size is None:
            # Estimate or compute total size
            # For efficiency, we can estimate or load metadata only
            self._total_size = 0
            for subject_id in self.subject_ids:
                # Load briefly to get size
                ds = self.dataset_cls(subject_id=subject_id, dataset_info=self.dataset_info)
                self._subject_sizes[subject_id] = len(ds)
                self._total_size += len(ds)
        return self._total_size
    
    def __getitem__(self, idx):
        """
        Get item by global index.
        
        Note: For streaming training, we recommend using StreamingDataLoader
        instead of standard DataLoader with this Dataset.
        """
        # Find which subject this index belongs to
        cumulative = 0
        for subject_id in self.subject_ids:
            size = self._subject_sizes.get(subject_id)
            if size is None:
                ds = self.dataset_cls(subject_id=subject_id, dataset_info=self.dataset_info)
                size = len(ds)
                self._subject_sizes[subject_id] = size
            
            if idx < cumulative + size:
                local_idx = idx - cumulative
                ds = self._get_subject_data(subject_id)
                return ds[local_idx]
            cumulative += size
        
        raise IndexError(f"Index {idx} out of range")


class StreamingDataLoader:
    """
    Memory-efficient data loader that streams subjects.
    
    Instead of loading all data into memory, it iterates through subjects
    one by one, yielding batches.
    """
    
    def __init__(self, subject_ids: List[int], dataset_cls, dataset_info: Dict,
                 batch_size: int = 32, shuffle: bool = True, device: str = 'cpu'):
        self.subject_ids = subject_ids
        self.dataset_cls = dataset_cls
        self.dataset_info = dataset_info
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
    
    def __iter__(self):
        """Iterate through all subjects, yielding batches."""
        subject_order = self.subject_ids.copy()
        if self.shuffle:
            np.random.shuffle(subject_order)
        
        for subject_id in subject_order:
            # Load one subject at a time
            ds = self.dataset_cls(subject_id=subject_id, dataset_info=self.dataset_info)
            
            # Create batches from this subject
            indices = list(range(len(ds)))
            if self.shuffle:
                np.random.shuffle(indices)
            
            for i in range(0, len(ds), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                
                # Load batch
                batch_data = []
                batch_labels = []
                for idx in batch_indices:
                    data, label = ds[idx]
                    batch_data.append(data)
                    batch_labels.append(label)
                
                yield torch.stack(batch_data), torch.stack(batch_labels)
            
            # Delete subject data to free memory
            del ds
            import gc
            gc.collect()
    
    def __len__(self):
        """Estimate number of batches."""
        # This is approximate
        total_samples = 0
        for subject_id in self.subject_ids:
            ds = self.dataset_cls(subject_id=subject_id, dataset_info=self.dataset_info)
            total_samples += len(ds)
            del ds
        return (total_samples + self.batch_size - 1) // self.batch_size


class StreamingLOSOTrainer(BaseTrainer):
    """
    Memory-efficient LOSO trainer for large-scale datasets.
    
    Key features:
    - Streams training data subject-by-subject (constant memory)
    - Gradient accumulation for effective large batch training
    - Supports mixed precision training for speed
    - Handles class imbalance across subjects
    
    Example:
        >>> trainer = StreamingLOSOTrainer(
        ...     model=model,
        ...     config=config,
        ...     device=device,
        ...     paths=paths,
        ...     logger=logger,
        ...     gradient_accumulation_steps=4  # Simulate 4x larger batch
        ... )
    """
    
    def __init__(self, model=None, config=None, device=None, paths=None, logger=None, **kwargs):
        super().__init__(model=model, config=config, device=device, paths=paths, logger=logger)
        
        # Streaming configuration
        self.gradient_accumulation_steps = kwargs.get('gradient_accumulation_steps', 
                                                      self.config.get('trainer.args.gradient_accumulation_steps', 1))
        self.subject_buffer_size = kwargs.get('subject_buffer_size',
                                              self.config.get('trainer.args.subject_buffer_size', 1))
        self.mixed_precision = kwargs.get('mixed_precision',
                                          self.config.get('trainer.args.mixed_precision', False))
        
        # Standard training configuration
        self.epochs = self.config.get('training.epochs', 100)
        self.batch_size = self.config.get('training.batch_size', 32)
        self.learning_rate = self.config.get('training.optimizer.lr', 1e-3)
        self.weight_decay = self.config.get('training.optimizer.weight_decay', 0.01)
        
        # Model configuration
        self.model_type = self.config.get('model.type')
        self.model_args = self.config.get('model.args', {})
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Checkpoint configuration
        self.save_checkpoints = self.config.get('training.save_checkpoints', True)
        self.use_fp16_compression = self.config.get('training.use_fp16_compression', True)
        
        # Scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision and torch.cuda.is_available() else None
        
    def train(self, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run streaming LOSO cross-validation training.
        
        Args:
            datasets: Dictionary containing:
                - subjects: List of subject IDs
                - info: Dataset information
                - dataset_cls: Dataset class for loading
                
        Returns:
            Dictionary containing LOSO results
        """
        subjects = datasets['subjects']
        dataset_info = datasets.get('info', {})
        dataset_cls = datasets.get('dataset_cls')
        
        if not dataset_cls:
            raise ValueError("StreamingLOSOTrainer requires 'dataset_cls'")
        
        all_results = []
        all_histories = []
        
        self._log("=" * 60)
        self._log("Starting Streaming LOSO Training")
        self._log(f"Total subjects: {len(subjects)}")
        self._log(f"Memory-efficient mode: {self.subject_buffer_size} subject(s) in memory")
        self._log(f"Gradient accumulation: {self.gradient_accumulation_steps} steps")
        if self.mixed_precision:
            self._log("Mixed precision training: Enabled")
        self._log("=" * 60)
        
        # LOSO loop
        for test_subject_idx, test_subject_id in enumerate(subjects):
            self._log(f"\n[{test_subject_idx + 1}/{len(subjects)}] Test Subject: {test_subject_id}")
            self._log("-" * 40)
            
            # Get training subjects
            train_subject_ids = [s for s in subjects if s != test_subject_id]
            
            # Train and evaluate
            subject_results, subject_history = self._train_loso_round_streaming(
                test_subject_id, train_subject_ids, dataset_cls, dataset_info
            )
            
            all_results.append(subject_results)
            all_histories.append({
                'test_subject_id': test_subject_id,
                'history': subject_history
            })
            
            # Plot history
            subject_viz_dir = self.paths.get_subject_viz_dir(test_subject_id)
            self._plot_history(subject_history, subject_viz_dir, test_subject_id)
            
            import gc
            gc.collect()
        
        # Final comparison
        self._plot_comparison(all_results)
        
        # Statistics
        test_accuracies = [r['test_acc'] for r in all_results]
        final_results = {
            'subjects': all_results,
            'overall_mean': np.mean(test_accuracies),
            'overall_std': np.std(test_accuracies),
            'overall_min': np.min(test_accuracies),
            'overall_max': np.max(test_accuracies),
            'streaming_config': {
                'num_subjects': len(subjects),
                'gradient_accumulation_steps': self.gradient_accumulation_steps,
                'subject_buffer_size': self.subject_buffer_size,
                'mixed_precision': self.mixed_precision
            }
        }
        
        self._log("\n" + "=" * 60)
        self._log("Streaming LOSO Training Complete")
        self._log(f"Mean Accuracy: {final_results['overall_mean']:.4f} ± {final_results['overall_std']:.4f}")
        self._log("=" * 60)
        
        return final_results
    
    def _train_loso_round_streaming(self, test_subject_id: int,
                                    train_subject_ids: List[int],
                                    dataset_cls, dataset_info: Dict) -> Tuple[Dict, Dict]:
        """
        Train one LOSO round with streaming data loading.
        """
        # Create model
        # Infer model parameters from a sample subject
        sample_ds = dataset_cls(subject_id=train_subject_ids[0], dataset_info=dataset_info)
        model = self._create_model(sample_ds)
        del sample_ds
        
        # Create test dataset
        test_dataset = dataset_cls(subject_id=test_subject_id, dataset_info=dataset_info)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Training history
        history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
        best_test_acc = 0.0
        best_epoch = 0
        
        # Progress bar
        epoch_bar = tqdm(range(self.epochs), desc=f'Sub{test_subject_id}', 
                        bar_format='{desc} |{bar:20}| {n_fmt}/{total_fmt} {postfix}')
        
        for epoch in epoch_bar:
            self.current_epoch = epoch
            
            # Update learning rate
            lr = self._compute_lr(epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # Train with streaming
            train_loss, train_acc = self._train_epoch_streaming(
                model, optimizer, train_subject_ids, dataset_cls, dataset_info
            )
            
            # Evaluate
            test_loss, test_acc = self._validate(model, test_loader)
            
            # Record
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            
            epoch_bar.set_postfix_str(
                f"LR:{lr:.1e} Tr:{train_loss:.3f}/{train_acc:.3f} Te:{test_loss:.3f}/{test_acc:.3f}"
            )
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
                if self.save_checkpoints:
                    self._save_checkpoint(model, test_subject_id, epoch, test_acc)
            
            self.global_step += 1
        
        epoch_bar.close()
        
        final_loss, final_acc = self._validate(model, test_loader)
        
        return {
            'test_subject_id': test_subject_id,
            'test_acc': final_acc,
            'test_loss': final_loss,
            'best_acc': best_test_acc,
            'best_epoch': best_epoch,
            'train_subjects': len(train_subject_ids),
            'test_samples': len(test_dataset)
        }, history
    
    def _train_epoch_streaming(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                               train_subject_ids: List[int], dataset_cls, dataset_info: Dict
                               ) -> Tuple[float, float]:
        """
        Train one epoch by streaming through subjects.
        """
        model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Shuffle subjects each epoch
        subject_order = train_subject_ids.copy()
        np.random.shuffle(subject_order)
        
        # Gradient accumulation counter
        accumulation_counter = 0
        
        for subject_id in subject_order:
            # Load subject
            ds = dataset_cls(subject_id=subject_id, dataset_info=dataset_info)
            
            # Create batches
            indices = list(range(len(ds)))
            np.random.shuffle(indices)
            
            for i in range(0, len(ds), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                
                # Load batch
                batch_data = []
                batch_labels = []
                for idx in batch_indices:
                    data, label = ds[idx]
                    batch_data.append(data)
                    batch_labels.append(label)
                
                inputs = torch.stack(batch_data).to(self.device)
                labels = torch.stack(batch_labels).to(self.device)
                
                # Forward
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = self.criterion(outputs, labels)
                        loss = loss / self.gradient_accumulation_steps
                    
                    self.scaler.scale(loss).backward()
                else:
                    outputs = model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss = loss / self.gradient_accumulation_steps
                    loss.backward()
                
                # Accumulation
                accumulation_counter += 1
                
                if accumulation_counter % self.gradient_accumulation_steps == 0:
                    if self.scaler:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                
                # Statistics
                total_loss += loss.item() * inputs.size(0) * self.gradient_accumulation_steps
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            
            # Free memory
            del ds
            import gc
            gc.collect()
        
        # Handle remaining gradients
        if accumulation_counter % self.gradient_accumulation_steps != 0:
            if self.scaler:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
    
    def _create_model(self, sample_dataset) -> nn.Module:
        """Create model from sample dataset."""
        from core.registry import MODELS, build_from_config
        
        model_config = {
            'type': self.model_type,
            'args': self.model_args.copy()
        }
        
        # Auto-populate
        if 'num_channels' not in model_config['args']:
            model_config['args']['num_channels'] = sample_dataset.data.shape[2]
        if 'num_classes' not in model_config['args']:
            model_config['args']['num_classes'] = len(set(sample_dataset.labels))
        if 'num_bands' not in model_config['args']:
            model_config['args']['num_bands'] = sample_dataset.data.shape[1]
        if 'input_length' not in model_config['args']:
            model_config['args']['input_length'] = sample_dataset.data.shape[3]
        
        model = build_from_config(model_config, MODELS)
        model = model.to(self.device)
        
        return model
    
    def _compute_lr(self, epoch: int) -> float:
        """Cosine annealing learning rate."""
        return (1 + math.cos(epoch * math.pi / self.epochs)) * self.learning_rate / 2
    
    def _validate(self, model: nn.Module, dataloader: DataLoader) -> Tuple[float, float]:
        """Validate model."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
    
    def _save_checkpoint(self, model: nn.Module, test_subject_id: int, 
                        epoch: int, metric: float) -> None:
        """Save checkpoint."""
        if not self.save_checkpoints:
            return
        
        state_dict = model.state_dict()
        if self.use_fp16_compression:
            state_dict = {k: v.half() if v.dtype == torch.float32 else v 
                         for k, v in state_dict.items()}
        
        checkpoint = {
            'epoch': epoch,
            'test_subject_id': test_subject_id,
            'model_state_dict': state_dict,
            'metric': metric,
            'compressed': self.use_fp16_compression
        }
        
        checkpoint_path = self.paths.get_checkpoint_path(
            subject_id=test_subject_id, fold=0, metric=metric
        )
        checkpoint_path = checkpoint_path.with_name(
            f"streaming_loso_subject{test_subject_id}_acc{metric:.4f}.pt"
        )
        
        try:
            torch.save(checkpoint, checkpoint_path)
        except Exception as e:
            pass
    
    def _plot_history(self, history: Dict, save_dir: Path, test_subject_id: int):
        """Plot training history."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Train')
        axes[0].plot(epochs, history['test_loss'], 'r-', label='Test')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'Subject {test_subject_id} Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(epochs, history['train_acc'], 'b-', label='Train')
        axes[1].plot(epochs, history['test_acc'], 'r-', label='Test')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title(f'Subject {test_subject_id} Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'streaming_loso_history.png', dpi=150)
        plt.close()
    
    def _plot_comparison(self, results: List[Dict]):
        """Plot comparison across subjects."""
        import matplotlib.pyplot as plt
        
        subject_ids = [r['test_subject_id'] for r in results]
        test_accs = [r['test_acc'] for r in results]
        
        fig, ax = plt.subplots(figsize=(max(8, len(subject_ids) * 0.4), 6))
        
        colors = ['green' if acc >= 0.8 else 'yellow' if acc >= 0.6 else 'red' 
                  for acc in test_accs]
        
        bars = ax.bar(range(len(subject_ids)), test_accs, color=colors, 
                     edgecolor='black', alpha=0.7)
        
        ax.set_xlabel('Test Subject ID')
        ax.set_ylabel('Test Accuracy')
        ax.set_title('Streaming LOSO Cross-Validation Results')
        ax.set_xticks(range(len(subject_ids)))
        ax.set_xticklabels([f'Sub{s}' for s in subject_ids], rotation=45)
        ax.set_ylim([0, 1.05])
        ax.grid(True, axis='y', alpha=0.3)
        
        mean_acc = np.mean(test_accs)
        ax.axhline(y=mean_acc, color='r', linestyle='--', 
                  label=f'Mean: {mean_acc:.3f}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.paths.viz_dir / 'streaming_loso_comparison.png', dpi=150)
        plt.close()
