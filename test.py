# src/datasets/base_dataset.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.factory import ProcessorFactory
from preprocessing.base import ProcessingPipeline


class BaseBCIDataset(Dataset, ABC):
    """
    Abstract base class for all BCI datasets in the platform.
    Provides common functionality for data loading and preprocessing.
    """
    
    def __init__(self, 
                 subject_id: int,
                 dataset_config: Dict[str, Any],
                 mode: str = 'train',
                 global_pipeline_config: Optional[list] = None,
                 sample_pipeline_config: Optional[list] = None,
                 augmentation_config: Optional[list] = None):
        """
        Args:
            subject_id: Subject identifier.
            dataset_config: Dataset-specific configuration.
            mode: One of ['train', 'val', 'test'].
            global_pipeline_config: Configuration for global preprocessing pipeline.
            sample_pipeline_config: Configuration for per-sample preprocessing.
            augmentation_config: Configuration for data augmentation (train mode only).
        """
        self.subject_id = subject_id
        self.dataset_config = dataset_config
        self.mode = mode
        
        # Validate mode
        valid_modes = ['train', 'val', 'test']
        if mode not in valid_modes:
            raise ValueError(f"Mode must be one of {valid_modes}, got {mode}")
        
        # Build preprocessing pipelines
        self.global_pipeline = self._build_global_pipeline(global_pipeline_config)
        self.sample_pipeline = self._build_sample_pipeline(sample_pipeline_config)
        self.augmentation_pipeline = self._build_augmentation_pipeline(augmentation_config)
        
        # Load and preprocess data
        self.data, self.labels, self.metadata = self._load_and_preprocess()
        
    def _build_global_pipeline(self, config: Optional[list]) -> Optional[ProcessingPipeline]:
        """Build global preprocessing pipeline (applied once per subject)."""
        if config is None:
            return None
        
        try:
            pipeline = ProcessorFactory.create_pipeline_from_config(config)
            print(f"[Dataset] Created global pipeline: {pipeline}")
            return pipeline
        except Exception as e:
            raise RuntimeError(f"Failed to build global pipeline: {e}")
    
    def _build_sample_pipeline(self, config: Optional[list]) -> Optional[ProcessingPipeline]:
        """Build per-sample preprocessing pipeline."""
        if config is None:
            return None
        
        try:
            pipeline = ProcessorFactory.create_pipeline_from_config(config)
            print(f"[Dataset] Created sample pipeline: {pipeline}")
            return pipeline
        except Exception as e:
            raise RuntimeError(f"Failed to build sample pipeline: {e}")
    
    def _build_augmentation_pipeline(self, config: Optional[list]) -> Optional[ProcessingPipeline]:
        """Build data augmentation pipeline (only for training mode)."""
        if self.mode != 'train' or config is None:
            return None
        
        try:
            pipeline = ProcessorFactory.create_pipeline_from_config(config)
            print(f"[Dataset] Created augmentation pipeline: {pipeline}")
            return pipeline
        except Exception as e:
            raise RuntimeError(f"Failed to build augmentation pipeline: {e}")
    
    def _apply_global_preprocessing(self, raw_data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply global preprocessing pipeline to raw data.
        
        Args:
            raw_data: Raw data loaded from disk.
            **kwargs: Additional parameters for processors (e.g., sample rate).
            
        Returns:
            Globally preprocessed data.
        """
        if self.global_pipeline is None:
            return raw_data
        
        # Add dataset-specific parameters to kwargs
        runtime_kwargs = {
            'sample_rate': self.dataset_config.get('sample_rate'),
            'subject_id': self.subject_id,
            'mode': self.mode,
            **kwargs
        }
        
        return self.global_pipeline(raw_data, **runtime_kwargs)
    
    def _apply_sample_preprocessing(self, sample: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply per-sample preprocessing pipeline.
        
        Args:
            sample: Individual data sample.
            **kwargs: Additional parameters for processors.
            
        Returns:
            Preprocessed sample ready for model input.
        """
        if self.sample_pipeline is None:
            return sample
        
        runtime_kwargs = {
            'sample_rate': self.dataset_config.get('sample_rate'),
            'mode': self.mode,
            **kwargs
        }
        
        return self.sample_pipeline(sample, **runtime_kwargs)
    
    def _apply_augmentation(self, sample: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply data augmentation to a sample (training mode only).
        
        Args:
            sample: Preprocessed data sample.
            **kwargs: Additional parameters for augmentation.
            
        Returns:
            Augmented sample.
        """
        if self.mode != 'train' or self.augmentation_pipeline is None:
            return sample
        
        runtime_kwargs = {
            'sample_rate': self.dataset_config.get('sample_rate'),
            **kwargs
        }
        
        return self.augmentation_pipeline(sample, **runtime_kwargs)
    
    @abstractmethod
    def _load_raw_data(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Abstract method to load raw data from disk.
        Must be implemented by concrete dataset classes.
        
        Returns:
            tuple: (data, labels, metadata)
        """
        pass
    
    def _load_and_preprocess(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Main pipeline: load raw data, apply global preprocessing, and prepare samples.
        
        Returns:
            tuple: (preprocessed_data, labels, metadata)
        """
        # 1. Load raw data
        raw_data, labels, metadata = self._load_raw_data()
        print(f"[Subject {self.subject_id}] Loaded raw data shape: {raw_data.shape}")
        
        # 2. Apply global preprocessing
        preprocessed_data = self._apply_global_preprocessing(
            raw_data, 
            original_shape=raw_data.shape
        )
        print(f"[Subject {self.subject_id}] After global preprocessing shape: {preprocessed_data.shape}")
        
        # 3. Apply sliding window segmentation (if needed)
        windowed_data, windowed_labels, group_ids = self._apply_sliding_window(
            preprocessed_data, labels
        )
        
        print(f"[Subject {self.subject_id}] Final dataset: {len(windowed_data)} windows")
        
        return windowed_data, windowed_labels, {
            'group_ids': group_ids,
            'original_metadata': metadata
        }
    
    def _apply_sliding_window(self, data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply sliding window segmentation to continuous data.
        This is a generic implementation that can be overridden by specific datasets.
        """
        # Extract parameters from config
        window_len_samples = int(self.dataset_config.get('window_length_sec', 3) * 
                               self.dataset_config.get('sample_rate', 250))
        window_stride_samples = int(self.dataset_config.get('window_stride_sec', 0.5) * 
                                  self.dataset_config.get('sample_rate', 250))
        
        # Implementation of sliding window...
        # (This would contain the actual windowing logic from your original code)
        
        return windowed_data, windowed_labels, group_ids
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset with all preprocessing applied.
        """
        # Get raw sample
        sample = self.data[idx]
        label = self.labels[idx]
        
        # Apply sample-level preprocessing
        sample = self._apply_sample_preprocessing(sample)
        
        # Apply data augmentation (only in training mode)
        sample = self._apply_augmentation(sample)
        
        # Convert to PyTorch tensor
        sample_tensor = torch.from_numpy(sample).float()
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return sample_tensor, label_tensor
    
    def get_metadata(self, idx: int) -> Dict:
        """Get metadata for a specific sample."""
        return {
            'subject_id': self.subject_id,
            'group_id': self.metadata['group_ids'][idx],
            'original_metadata': self.metadata['original_metadata']
        }