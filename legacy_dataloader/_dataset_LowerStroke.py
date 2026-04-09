"""
Dataset loader for the "Lower limb motor imagery EEG dataset based on the multi-paradigm and longitudinal-training of stroke patients"
Publication: Scientific Data, 2025 (DOI: https://doi.org/10.1038/s41597-025-04618-4)
Dataset Structure: Organized in BIDS format, containing '.set' files (EEGLAB format) and patient info.
Revised implementation based on test.py pattern for reading preprocessed PEEG data.
"""

import mne
import os
import numpy as np
import pandas as pd
from legacy_dataloader._dataset_Base import BaseDataset
from typing import List, Tuple, Dict, Any, Optional
import warnings


class StrokeLowerLimbMI(BaseDataset):
    """
    Dataset class for the Stroke Lower Limb MI EEG Dataset.
    
    This implementation follows the pattern from test.py for reading BIDS-formatted
    preprocessed EEG data. It properly handles the event structure and extracts
    epochs based on event markers.
    
    Key features from paper and data structure:
    - Subjects: 27 stroke patients
    - Paradigms: Pre (Text), IES, SES, Post, Follow-up
    - Sessions: Each paradigm has session 'ses-1'
    - File naming: sub-{xx}_{paradigm}_run-{y}_eeg.set
    - Data organization: BIDS format with PEEG folder structure
    - Events: Labels 1-12 marking different phases
    """
    
    def _find_data_file(self, subject_id_str: str, paradigm: str, run_num: str = '1') -> str:
        """
        Find the EEG data file following the BIDS structure.
        
        Args:
            subject_id_str: Subject ID string (e.g., 'sub-01')
            paradigm: Experiment paradigm (e.g., 'Pre', 'IES', 'SES', 'Post', 'Follow')
            run_num: Run number, typically '1'
            
        Returns:
            Path to the EEG data file
            
        Raises:
            FileNotFoundError: If no valid data file is found
        """
        # Construct the expected file name
        file_name = f'{subject_id_str}_{paradigm.lower()}_run-{run_num}_eeg.set'
        
        # Try the main PEEG directory structure first
        main_path = os.path.join(
            self.data_dir,  # This should point to the PEEG folder
            subject_id_str,
            "ses-1",
            "eeg", 
            file_name
        )
        
        if os.path.exists(main_path):
            return main_path
        
        # If not found, try alternative paths
        alt_paths = [
            # Try derivatives folder
            os.path.join(self.data_dir, 'derivatives', subject_id_str, file_name),
            # Try sourcedata folder (original data)
            os.path.join(self.data_dir, 'sourcedata', subject_id_str, 
                        file_name.replace('_eeg.set', '_ori.set')),
            # Try direct location
            os.path.join(self.data_dir, subject_id_str, 
                        file_name.replace('_eeg.set', '_ori.set'))
        ]
        
        for path in alt_paths:
            if os.path.exists(path):
                print(f"[INFO] Found file at alternative path: {path}")
                return path
        
        raise FileNotFoundError(
            f"No EEG data file found for subject {subject_id_str}, paradigm {paradigm}. "
            f"Tried paths:\n- {main_path}\n" + "\n- ".join(alt_paths)
        )
    
    def _extract_epochs_from_raw(self, raw, event_mapping: Dict[str, int], 
                                 tmin: float = 0, tmax: float = 4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract epochs from raw EEG data based on event markers.
        
        Args:
            raw: MNE Raw object containing EEG data
            event_mapping: Dictionary mapping event descriptions to class labels
            tmin: Start time before event (seconds)
            tmax: End time after event (seconds)
            
        Returns:
            Tuple of (epochs_data, epochs_labels) where:
                epochs_data: EEG epochs array of shape (n_epochs, n_channels, n_times)
                epochs_labels: Label array of shape (n_epochs,)
        """
        # Extract events from annotations
        if raw.annotations is None or len(raw.annotations) == 0:
            warnings.warn(f"No annotations found in the EEG data")
            return np.array([]), np.array([])
        
        # Convert annotations to events
        events, event_dict = mne.events_from_annotations(raw)
        
        if len(events) == 0:
            warnings.warn(f"No events found in the EEG data")
            return np.array([]), np.array([])
        
        print(f"[DEBUG] Found {len(events)} events")
        print(f"[DEBUG] Event mapping: {event_dict}")
        
        # Filter events based on our mapping
        valid_events = []
        valid_event_ids = {}
        
        for event_desc, class_label in event_mapping.items():
            if event_desc in event_dict:
                event_id = event_dict[event_desc]
                valid_event_ids[event_desc] = event_id
                # Find all events of this type
                event_indices = np.where(events[:, 2] == event_id)[0]
                valid_events.extend(events[event_indices])
        
        if not valid_events:
            warnings.warn(f"No valid events found for mapping {event_mapping}")
            return np.array([]), np.array([])
        
        valid_events = np.array(valid_events)
        
        # Create epochs
        epochs = mne.Epochs(
            raw, 
            valid_events, 
            event_id=valid_event_ids,
            tmin=tmin, 
            tmax=tmax,
            baseline=None,
            preload=True,
            verbose=False
        )
        
        # Extract data and create labels
        epochs_data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
        
        # Create labels based on event mapping
        epochs_labels = []
        for event in valid_events:
            event_id = event[2]
            # Find which event description this corresponds to
            for event_desc, class_label in event_mapping.items():
                if event_desc in event_dict and event_dict[event_desc] == event_id:
                    epochs_labels.append(class_label)
                    break
        
        epochs_labels = np.array(epochs_labels, dtype=np.int64)
        
        return epochs_data, epochs_labels
    
    def _load_paradigm_data(self, subject_id_str: str, paradigm: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load EEG data for a specific subject and paradigm.
        
        Args:
            subject_id_str: Subject ID string
            paradigm: Experiment paradigm
            
        Returns:
            Tuple of (eeg_data, labels) for the paradigm
        """
        try:
            # Find and load the EEG file
            file_path = self._find_data_file(subject_id_str, paradigm)
            print(f"[INFO] Loading data from: {file_path}")
            
            # Load raw EEG data
            raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
            
            # Get basic information
            sfreq = raw.info['sfreq']
            n_channels = len(raw.info['ch_names'])
            print(f"[INFO] Loaded EEG data: {sfreq} Hz, {n_channels} channels")
            
            # Define event mapping based on paper
            # According to the paper: labels 1-6 are gait MI, 7-12 are idle state
            # We need to map these to binary classes
            event_mapping = {}
            
            # Get configuration for label mapping
            mi_labels = self.info['dataset'].get('mi_event_labels', ['1', '2', '3', '4', '5', '6'])
            idle_labels = self.info['dataset'].get('idle_event_labels', ['7', '8', '9', '10', '11', '12'])
            
            # Create mapping from event descriptions to class labels
            # MI events -> class 1, Idle events -> class 0
            for label in mi_labels:
                event_mapping[label] = 1
            for label in idle_labels:
                event_mapping[label] = 0
            
            # Define epoch time window
            # Based on paper: MI phase is 4-8s, idle phase is 4s
            # We'll use 4 seconds as a standard window
            tmin = self.info['preprocessing'].get('epoch_tmin', 0)
            tmax = self.info['preprocessing'].get('epoch_tmax', 4)
            
            # Extract epochs
            epochs_data, epochs_labels = self._extract_epochs_from_raw(
                raw, event_mapping, tmin, tmax
            )
            
            if len(epochs_data) == 0:
                print(f"[WARNING] No epochs extracted for {subject_id_str}, paradigm {paradigm}")
                return np.array([]), np.array([])
            
            print(f"[INFO] Extracted {len(epochs_data)} epochs for {subject_id_str}, paradigm {paradigm}")
            
            
            splited_data = []
            for epoch in epochs_data:
                normalized_data = self._exponential_moving_standardize(
                    epoch,
                    init_block_size=int(raw.info['sfreq'] * 4)
                )
                splited_data.append(normalized_data)
            session_data = np.stack(splited_data)
    
            return session_data, epochs_labels
            
        except FileNotFoundError as e:
            print(f"[WARNING] {e}")
            return np.array([]), np.array([])
        except Exception as e:
            print(f"[ERROR] Error loading data for {subject_id_str}, paradigm {paradigm}: {str(e)}")
            return np.array([]), np.array([])
    
    def _load_raw_data(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Load raw EEG data for the specified subject from all available paradigms.
        
        Returns:
            Tuple of (session_data_list, label_list) where:
                session_data_list: List of EEG data arrays for each session/paradigm
                label_list: List of label arrays for each session/paradigm
                
        Raises:
            FileNotFoundError: If no valid data is found for the subject
        """
        subject_id_str = f"sub-{self.subject_id:02d}"
        
        # Get paradigms to load from configuration
        paradigms_to_load = self.info['dataset'].get('paradigms', ['Pre', 'IES', 'SES', 'Post', 'Follow'])
        
        # Convert to lowercase to match file naming
        paradigms_to_load = [p.lower() for p in paradigms_to_load]
        
        session_data_list = []
        session_label_list = []
        loaded_paradigms = []
        
        # Load data for each paradigm
        for paradigm in paradigms_to_load:
            print(f"[INFO] Loading paradigm: {paradigm} for subject {subject_id_str}")
            
            eeg_data, labels = self._load_paradigm_data(subject_id_str, paradigm)
            
            if len(eeg_data) > 0 and len(labels) > 0:
                session_data_list.append(eeg_data)
                session_label_list.append(labels)
                loaded_paradigms.append(paradigm)
                print(f"[INFO] Successfully loaded {len(eeg_data)} trials from paradigm {paradigm}")
        
        # Check if any data was loaded
        if not session_data_list:
            error_msg = (
                f"No valid EEG data found for subject {self.subject_id} in paradigms: {paradigms_to_load}.\n"
                f"Possible reasons:\n"
                f"1. Data path is incorrect: {self.data_dir}\n"
                f"2. Subject {self.subject_id} did not participate in these paradigms (check Table 1 in paper)\n"
                f"3. File naming doesn't match expected pattern: sub-{{id}}_{{paradigm}}_run-{{run}}_eeg.set\n"
                f"4. Data might be in a different location (check derivatives/ or sourcedata/ folders)"
            )
            raise FileNotFoundError(error_msg)
        
        print(f"[INFO] Successfully loaded data for subject {self.subject_id} from paradigms: {loaded_paradigms}")
        
        return session_data_list, session_label_list
    
    def _load_participant_info(self) -> Optional[Dict[str, Any]]:
        """
        Load participant information from participants.tsv file.
        
        Returns:
            Dictionary containing participant information, or None if not available
        """
        participants_tsv_path = os.path.join(self.data_dir, 'participants.tsv')
        
        if os.path.exists(participants_tsv_path):
            try:
                df_participants = pd.read_csv(participants_tsv_path, sep='\t')
                subject_info = df_participants[df_participants['participant_id'] == f'sub-{self.subject_id:02d}']
                
                if not subject_info.empty:
                    # Convert to dictionary
                    info_dict = subject_info.iloc[0].to_dict()
                    print(f"[INFO] Loaded participant info for subject {self.subject_id}")
                    return info_dict
                else:
                    print(f"[WARNING] No participant info found for subject {self.subject_id} in participants.tsv")
            except Exception as e:
                print(f"[WARNING] Could not load participant info: {str(e)}")
        
        return None
    
    def get_subject_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded subject.
        
        Returns:
            Dictionary containing subject information
        """
        info = {
            'subject_id': self.subject_id,
            'subject_str': f"sub-{self.subject_id:02d}",
            'num_sessions': len(self.data) if hasattr(self, 'data') else 0,
            'data_dir': self.data_dir
        }
        
        # Try to load participant info
        participant_info = self._load_participant_info()
        if participant_info:
            info.update(participant_info)
        
        return info