from torch.utils.data import Dataset
from preprocessing.slide_windows import SlidingWindowSegmenter
from preprocessing.filter_banks import FilterBankProcessor
# from braindecode.preprocessing import Preprocessor, preprocess, PreprocessingPipeline

class BaseDataset(Dataset):
    def __init__(self, 
                 subject_id: int, dataset_info: dict, transform=None):
        self.subject_id = subject_id
        self.transform = transform

        # Load dataset metadata from the YAML configuration file
        self.info = dataset_info
        self.data_dir = self.info['dataset']['data_dir']

        self.channels_selected = self.info['preprocessing']['channel_selection']['channels_to_select']
        self.sample_rate = self.info['preprocessing']['resample']['target_sr']

        # Assuming constant_value.window_length and constant_value.window_stride are defined globally
        self.window_len_samples = int(self.info['preprocessing']["windowing"]['window_length_sec'] * self.sample_rate)
        self.window_stride_samples = int(self.info['preprocessing']["windowing"]['window_stride_sec'] * self.sample_rate)

        # Build preprocessing pipeline
        self.filter_banks = self.info['preprocessing']['filter_bank']['filter_banks']
        self.n_bands = len(self.filter_banks)

        # self.global_pipeline = self._build_global_pipeline(self.info['preprocessing']['global_pipeline'])
        # self.sample_pipeline = self._build_sample_pipeline(self.info['preprocessing']['sample_pipeline'])
        # self.augmentation_pipeline = self._build_augmentation_pipeline(self.info['preprocessing']['augmentation_pipeline'])

        # Load and process the subject's data
        self.data, self.labels, self.group_ids = self._load_and_process_subject_data()
        print(f"[{self.info['dataset']['name']} Subject {self.subject_id}] Data loaded. Final shape: {self.data.shape}, Labels: {self.labels.shape}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]  # 形状: (n_bands, n_channels, window_len)
        label = self.labels[idx]
        # group_id = self.group_ids[idx]  # 如果需要，可以返回

        if self.transform:
            data = self.transform(data)

        # 返回格式与XWStrokeDataset一致
        return data, label
    

    def _load_and_process_subject_data(self,):
         # 1. Construct file path and load raw data
        raw_eeg_data, raw_labels = self._load_raw_data()  # raw_eeg_data: (n_sessions, n_all_channels, n_timepoints)

        # 2. Channel selection
        selected_eeg_data = raw_eeg_data[:, self.channels_selected, :]  # (n_sessions, n_selected_channels, n_timepoints)

        # 3. Apply preprocessing pipeline (Filtering + Normalization)
        # processed_eeg_data = self._apply_preprocessing(selected_eeg_data)  # (n_sessions, n_selected_channels, n_timepoints)
        filer_bank_processor = FilterBankProcessor(filter_banks=self.filter_banks, sample_rate=self.sample_rate)
        processed_eeg_data = filer_bank_processor(selected_eeg_data)

        # 4. Label shift (1,2 to 0,1)
        if min(self.info['dataset']['labels']) == 1:
            adjusted_labels = raw_labels - 1
        else:
            adjusted_labels = raw_labels

        # 5. Sliding window segmentation
        
        windows_segmenter = SlidingWindowSegmenter(self.window_len_samples, self.window_stride_samples)
        final_data, final_labels, final_group_ids = windows_segmenter(processed_eeg_data, labels = adjusted_labels) # 基类中使用**keywords传递参数，所以需要显式说明labels参数名

        print(f"[Subject {self.subject_id}] Data loaded. Final data shape: {final_data.shape}, Label shape: {final_labels.shape}")
        return final_data, final_labels, final_group_ids
    
    # def _apply_preprocessing(self, eeg_data):


    def _load_raw_data(self, file_path):
        # This method should be implemented in the subclass to load raw data from the given file path
        raise NotImplementedError("Subclasses must implement _load_raw_data method")
    
