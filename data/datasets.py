"""
Dataset implementations for EEG-BCI experiments.

These classes wrap the existing dataset implementations to conform
to the new BaseDataset interface with registry support.
"""

import os
from typing import Tuple
import numpy as np
import scipy.io as sio

from core.base.base_dataset import BaseDataset


class BCICIV2aDataset(BaseDataset):
    """
    BCI Competition IV 2a dataset.
    
    Motor imagery dataset with 4 classes (left hand, right hand, foot, tongue)
    from 9 subjects.
    """
    
    def _load_raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load raw data from .mat files.
        
        Returns:
            Tuple of (data, labels)
        """
        data_dir = self.data_dir or 'src/datasets/BCICIV_2a'
        
        # Load training data
        train_file = os.path.join(data_dir, f'A{self.subject_id:02d}T.mat')
        train_data = sio.loadmat(train_file)
        
        # Load test data
        test_file = os.path.join(data_dir, f'A{self.subject_id:02d}E.mat')
        test_data = sio.loadmat(test_file)
        
        # Extract data
        train_data_array = train_data['data']  # Shape: (n_channels, n_timepoints, n_trials)
        train_labels = train_data['labels'].flatten()
        
        test_data_array = test_data['data']
        test_labels = test_data['labels'].flatten()
        
        # Concatenate train and test
        combined_data = np.concatenate([train_data_array, test_data_array], axis=2)
        combined_labels = np.concatenate([train_labels, test_labels])
        
        # Transpose to (n_trials, n_channels, n_timepoints)
        combined_data = np.transpose(combined_data, (2, 0, 1))
        
        return combined_data, combined_labels


class XWStrokeDataset(BaseDataset):
    """
    XuanWu Stroke dataset.
    
    Stroke patient motor imagery dataset.
    Supports both original .mat files and preprocessed _4s.mat files.
    """
    
    def _load_raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load raw data from .mat files.
        
        Supports two modes:
        1. Original data: sub-XX_task-motor-imagery_eeg.mat
        2. 4s preprocessed data: sub-XX_task-motor-imagery_eeg_4s.mat
        
        Returns:
            Tuple of (data, labels)
        """
        # Get configuration
        data_dir = self.dataset_info.get('dataset', {}).get('data_dir', 'src/datasets/21679035/sourcedata')
        use_4s_data = self.dataset_info.get('dataset', {}).get('use_4s_data', True)
        data_var_name = self.dataset_info.get('dataset', {}).get('data_var_name', 'eeg_data_4s')
        labels_var_name = self.dataset_info.get('dataset', {}).get('labels_var_name', 'labels')
        
        # Build file path
        subject_dir = os.path.join(data_dir, f'sub-{self.subject_id:02d}')
        
        if use_4s_data:
            # Try to load 4s preprocessed data
            subject_file = os.path.join(subject_dir, f'sub-{self.subject_id:02d}_task-motor-imagery_eeg_4s.mat')
            
            if os.path.exists(subject_file):
                print(f"[Subject {self.subject_id}] Loading 4s preprocessed data: {subject_file}")
                mat_data = sio.loadmat(subject_file)
                
                # Extract data from 4s file
                if data_var_name in mat_data:
                    data = mat_data[data_var_name]
                    labels = mat_data[labels_var_name].flatten()
                    print(f"[Subject {self.subject_id}] Loaded 4s data shape: {data.shape}, labels shape: {labels.shape}")
                    return data, labels
                else:
                    print(f"[Subject {self.subject_id}] Warning: '{data_var_name}' not found in {subject_file}, trying original file...")
            else:
                print(f"[Subject {self.subject_id}] 4s file not found: {subject_file}, trying original file...")
        
        # Fallback to original file
        subject_file = os.path.join(subject_dir, f'sub-{self.subject_id:02d}_task-motor-imagery_eeg.mat')
        
        if not os.path.exists(subject_file):
            # Try alternative path structure (without sub-XX subdirectory)
            subject_file = os.path.join(data_dir, f'sub-{self.subject_id:02d}_task-motor-imagery_eeg.mat')
        
        print(f"[Subject {self.subject_id}] Loading original data: {subject_file}")
        mat_data = sio.loadmat(subject_file)
        
        # Extract data from original file
        if 'data' in mat_data:
            data = mat_data['data']
            labels = mat_data['labels'].flatten()
        elif 'eeg_data' in mat_data:
            data = mat_data['eeg_data']
            labels = mat_data['eeg_labels'].flatten()
        else:
            # Try to infer from structure
            data = None
            labels = None
            for key in mat_data.keys():
                if not key.startswith('__'):
                    if isinstance(mat_data[key], np.ndarray):
                        if len(mat_data[key].shape) == 3 and data is None:
                            data = mat_data[key]
                        elif len(mat_data[key].shape) == 1 and labels is None and 'label' in key.lower():
                            labels = mat_data[key]
            
            if data is None:
                raise ValueError(f"[Subject {self.subject_id}] Could not find data array in {subject_file}")
            if labels is None:
                raise ValueError(f"[Subject {self.subject_id}] Could not find labels array in {subject_file}")
        
        print(f"[Subject {self.subject_id}] Loaded original data shape: {data.shape}, labels shape: {labels.shape}")
        
        # Ensure correct shape (n_trials, n_channels, n_timepoints)
        if data.shape[1] < data.shape[2]:
            # Likely (n_trials, n_timepoints, n_channels), need to transpose
            data = np.transpose(data, (0, 2, 1))
            print(f"[Subject {self.subject_id}] Transposed data to: {data.shape}")
        
        return data, labels


class LowerStrokeDataset(BaseDataset):
    """
    Lower Limb Stroke dataset.
    
    Lower limb motor imagery dataset for stroke patients.
    """
    
    def _load_raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load raw data from preprocessed files.
        
        Returns:
            Tuple of (data, labels)
        """
        data_dir = self.data_dir or 'src/datasets/LowerStroke'
        
        # Load preprocessed data
        subject_file = os.path.join(data_dir, f'subject_{self.subject_id}.mat')
        
        mat_data = sio.loadmat(subject_file)
        
        # Extract data
        if 'data' in mat_data:
            data = mat_data['data']
            labels = mat_data['labels'].flatten()
        else:
            # Handle alternative formats
            data = mat_data[list(mat_data.keys())[-1]]
            labels = np.zeros(data.shape[0])  # Placeholder
        
        # Ensure correct shape
        if len(data.shape) == 2:
            # Single trial, add batch dimension
            data = np.expand_dims(data, axis=0)
        elif data.shape[1] < data.shape[2]:
            data = np.transpose(data, (0, 2, 1))
        
        return data, labels
