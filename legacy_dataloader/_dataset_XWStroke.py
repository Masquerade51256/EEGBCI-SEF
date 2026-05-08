import scipy.io
import os
from legacy_dataloader._dataset_Base import BaseDataset

class XWStrokeDataset(BaseDataset):
    def _load_mat_data(self, file_path):
        """Loads raw EEG data and labels from a .mat file."""
        try:
            data = scipy.io.loadmat(file_path)

            # for key, value in data.items():
            #     if not key.startswith('__'):
            #         eeg_data = value[0]  # Expected shape: (n_sessions, n_channels, n_timepoints)
            #         labels = value[1]    # Expected shape: (n_sessions, 1)
            #         break

            eeg_data = data['eeg_data_4s']
            labels = data['labels']
            return eeg_data, labels.reshape(-1)  # Flatten labels to (n_sessions,)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {file_path}")
        except KeyError:
            raise KeyError(f"File structure does not match expectations: {file_path}")

    def _load_raw_data(self):
        """Constructs the file path for the subject's data based on the dataset info."""
        file_name = f"sub-{self.subject_id:02d}_task-motor-imagery_eeg_4s.mat"
        file_path = os.path.join(self.data_dir, f"sub-{self.subject_id:02d}", file_name)
        data, labels = self._load_mat_data(file_path)
        
        # Optional: remap left/right labels to affected/unaffected
        label_mapping = self.info.get('dataset', {}).get('label_mapping', 'lr')
        if label_mapping != 'lr':
            import sys
            project_root = os.path.join(os.path.dirname(__file__), '..')
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            from data.datasets import _load_paralysis_side
            paralysis_side = _load_paralysis_side(self.subject_id, self.info)
            
            # Current labels from .mat are 1-based: 1=left, 2=right
            if label_mapping == 'affected':
                # Target: 0=affected, 1=unaffected
                if paralysis_side == 'left':
                    labels = labels - 1  # 1->0 (left/affected), 2->1 (right/unaffected)
                else:  # 'right'
                    labels = 2 - labels  # 1->1 (left/unaffected), 2->0 (right/affected)
            elif label_mapping == 'unaffected_first':
                # Target: 0=unaffected, 1=affected
                if paralysis_side == 'left':
                    labels = 2 - labels  # 1->1 (left/affected), 2->0 (right/unaffected)
                else:  # 'right'
                    labels = labels - 1  # 1->0 (left/unaffected), 2->1 (right/affected)
            else:
                raise ValueError(f"Unknown label_mapping: {label_mapping}")
            
            # Tell BaseDataset that labels are already 0-based so it won't subtract 1 again
            self.info['dataset']['labels'] = [0, 1]
        
        normalized_data = []
        for sess_data in data:
            sess_data = self._exponential_moving_standardize(sess_data, init_block_size=self.sample_rate)
            normalized_data.append(sess_data)
        return normalized_data, labels
