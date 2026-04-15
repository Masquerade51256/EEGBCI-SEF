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
    
    Note: Data is loaded from GDF files using MNE, consistent with torch_fold_train.py.
    """
    
    def _load_raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load raw data from GDF files using MNE.
        
        Returns:
            Tuple of (data, labels) where data shape is (n_trials, n_channels, n_timepoints)
        """
        import mne
        
        data_dir = self.data_dir or 'src/datasets/BCICIV2a'
        
        # Load training data (T file)
        train_gdf_file = os.path.join(data_dir, f'A{self.subject_id:02d}T.gdf')
        train_mat_file = os.path.join(data_dir, f'A{self.subject_id:02d}T.mat')
        
        # Load test data (E file)
        test_gdf_file = os.path.join(data_dir, f'A{self.subject_id:02d}E.gdf')
        test_mat_file = os.path.join(data_dir, f'A{self.subject_id:02d}E.mat')
        
        # Process training data
        train_data, train_labels = self._load_gdf_session(train_gdf_file, train_mat_file, is_train=True)
        
        # Process test data
        test_data, test_labels = self._load_gdf_session(test_gdf_file, test_mat_file, is_train=False)
        
        # Concatenate train and test data
        combined_data = np.concatenate([train_data, test_data], axis=0)
        combined_labels = np.concatenate([train_labels, test_labels])
        
        return combined_data, combined_labels
    
    def _load_gdf_session(self, gdf_file: str, mat_file: str, is_train: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a single session (train or test) from GDF file.
        
        Args:
            gdf_file: Path to GDF file
            mat_file: Path to MAT file (for labels)
            is_train: Whether this is training data
            
        Returns:
            Tuple of (data, labels)
        """
        import mne
        
        # Load GDF file
        raw = mne.io.read_raw_gdf(gdf_file, preload=True, verbose=False)
        events, annot = mne.events_from_annotations(raw)
        
        raw.load_data()
        raw.filter(0., 40., fir_design='firwin', verbose=False)
        raw.info['bads'] += ['EOG-left', 'EOG-central', 'EOG-right']
        
        picks = mne.pick_types(raw.info,
                            meg=False,
                            eeg=True,
                            eog=False,
                            stim=False,
                            exclude='bads')
        
        # Define event IDs and time window
        tmin, tmax = 0, 3  # 3 second epochs
        
        if is_train:
            if self.subject_id != 4:
                event_id = {'769': 7, '770': 8, '771': 9, '772': 10}
            else:
                event_id = {'769': 5, '770': 6, '771': 7, '772': 8}
        else:
            event_id = {'783': 7}
        
        epochs = mne.Epochs(raw,
                        events,
                        event_id,
                        tmin,
                        tmax,
                        proj=True,
                        picks=picks,
                        baseline=None,
                        preload=True,
                        verbose=False)
        
        # Get data (convert to microvolts)
        epochs_data = epochs.get_data() * 1e6  # Shape: (n_trials, n_channels, n_timepoints)
        
        # Load labels from mat file
        labels = sio.loadmat(mat_file)['classlabel'].reshape(-1) - 1  # Convert to 0-based
        
        return epochs_data, labels


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
                mat_data = sio.loadmat(subject_file)
                
                # Extract data from 4s file
                if data_var_name in mat_data:
                    data = mat_data[data_var_name]
                    labels = mat_data[labels_var_name].flatten()
                    return data, labels
            # Fall through to original file if 4s data not available
        
        # Fallback to original file
        subject_file = os.path.join(subject_dir, f'sub-{self.subject_id:02d}_task-motor-imagery_eeg.mat')
        
        if not os.path.exists(subject_file):
            # Try alternative path structure (without sub-XX subdirectory)
            subject_file = os.path.join(data_dir, f'sub-{self.subject_id:02d}_task-motor-imagery_eeg.mat')
        
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
        
        # Ensure correct shape (n_trials, n_channels, n_timepoints)
        if data.shape[1] < data.shape[2]:
            # Likely (n_trials, n_timepoints, n_channels), need to transpose
            data = np.transpose(data, (0, 2, 1))
        
        return data, labels


class XWStrokeEDFDataset(BaseDataset):
    """
    XuanWu Stroke dataset loaded from original EDF files.
    
    This dataset loader reads the raw EDF recordings and parses BIDS event
    information to extract the 4-second motor imagery (MI) epochs. It is
    designed to closely match the preprocessing pipeline described in
    Thwe et al. (2026), which starts from .edf files rather than
    pre-processed .mat files.
    
    Expected data layout:
        src/datasets/21679035/edffile/sub-XX/eeg/sub-XX_task-motor-imagery_eeg.edf
    Events are read from the shared BIDS TSV:
        src/datasets/21679035/task-motor-imagery_events.tsv
    """
    
    def _load_raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load raw EDF data and extract MI epochs.
        
        Returns:
            Tuple of (data, labels) where data shape is (n_trials, n_channels, n_timepoints)
        """
        import mne
        import pandas as pd
        import os
        
        # Get configuration
        data_dir = self.dataset_info.get('dataset', {}).get('data_dir', 'src/datasets/21679035')
        events_tsv = self.dataset_info.get('dataset', {}).get('events_tsv', 'task-motor-imagery_events.tsv')
        mi_value = self.dataset_info.get('dataset', {}).get('mi_event_value', 2)
        
        # Build EDF file path
        edf_file = os.path.join(
            data_dir, 'edffile', f'sub-{self.subject_id:02d}', 'eeg',
            f'sub-{self.subject_id:02d}_task-motor-imagery_eeg.edf'
        )
        
        if not os.path.exists(edf_file):
            raise FileNotFoundError(f"EDF file not found: {edf_file}")
        
        # Load EDF
        raw = mne.io.read_raw_edf(edf_file, preload=True, verbose='WARNING')
        
        # Remove empty-named channel if present (stim placeholder in this dataset)
        picks = [ch for ch in raw.ch_names if ch.strip()]
        raw.pick(picks)
        
        sfreq = raw.info['sfreq']
        
        # Read events TSV
        events_path = os.path.join(data_dir, events_tsv)
        if not os.path.exists(events_path):
            raise FileNotFoundError(f"Events TSV not found: {events_path}")
        
        events_df = pd.read_csv(events_path, sep='\t')
        
        # Filter MI events
        mi_events = events_df[events_df['value'] == mi_value].copy()
        if len(mi_events) == 0:
            raise ValueError(f"No MI events (value={mi_value}) found in {events_path}")
        
        # Extract epochs
        # The TSV uses milliseconds for onset and duration
        epochs_data = []
        labels = []
        
        for _, row in mi_events.iterrows():
            onset_sec = row['onset'] / 1000.0
            duration_sec = row['duration'] / 1000.0
            
            start_sample = int(round(onset_sec * sfreq))
            end_sample = start_sample + int(round(duration_sec * sfreq))
            
            epoch = raw.get_data(start=start_sample, stop=end_sample)  # (n_channels, n_times)
            epochs_data.append(epoch)
            labels.append(int(row['trial_type']))
        
        data = np.array(epochs_data)  # (n_trials, n_channels, n_times)
        labels = np.array(labels)
        
        # Convert from Volts to uV for consistency with typical EEG processing
        data = data * 1e6
        
        # Reorder channels to match standard XWStroke cap order if needed
        ch_names = [ch.upper() for ch in raw.ch_names]
        standard_order = [
            'FP1', 'FP2', 'FZ', 'F3', 'F4', 'F7', 'F8', 'FCZ', 'FC3', 'FC4',
            'FT7', 'FT8', 'CZ', 'C3', 'C4', 'T3', 'T4', 'CPZ', 'CP3', 'CP4',
            'TP7', 'TP8', 'PZ', 'P3', 'P4', 'T5', 'T6', 'OZ', 'O1', 'O2',
            'HEOL', 'HEOR'
        ]
        
        if set(ch_names) >= set(standard_order):
            reorder_idx = [ch_names.index(ch) for ch in standard_order if ch in ch_names]
            if len(reorder_idx) == data.shape[1]:
                data = data[:, reorder_idx, :]
        
        print(f"[Subject {self.subject_id}] Loaded EDF: {data.shape}, sfreq={sfreq}Hz, labels={labels[:5]}...")
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
