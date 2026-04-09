from torch.utils.data import Dataset
import torch
from preprocessing.slide_windows import SlidingWindowSegmenter
from preprocessing.filter_banks import FilterBankProcessor
from preprocessing.resample import ResampleProcessor
from preprocessing.data_augmentation import DataAugmentationProcessor
# from preprocessing.artifact_removal import ArtifactRemovalProcessor
import numpy as np
import pandas as pd

class BaseDataset(Dataset):
    def __init__(self, 
                 subject_id: int, dataset_info: dict, transform=None, mode="train"):
        self.subject_id = subject_id
        self.transform = transform
        self.mode = mode

        # Load dataset metadata from the YAML configuration file
        self.info = dataset_info
        self.data_dir = self.info['dataset']['data_dir']

        self.channels_selected = self.info['preprocessing']['channel_selection']['channels_to_select']
        self.original_sr = self.info['dataset']['original_sr']
        self.target_sr = self.info['preprocessing']['resample']['target_sr']
        self.sample_rate = self.original_sr

        # Build preprocessing pipeline
        self.filter_banks = self.info['preprocessing']['filter_bank']['filter_banks']
        self.n_bands = len(self.filter_banks)

        # self.global_pipeline = self._build_global_pipeline(self.info['preprocessing']['global_pipeline'])
        # self.sample_pipeline = self._build_sample_pipeline(self.info['preprocessing']['sample_pipeline'])
        # self.augmentation_pipeline = self._build_augmentation_pipeline(self.info['preprocessing']['augmentation_pipeline'])

        # Load and process the subject's data
        self.data, self.labels, self.group_ids = self._load_and_process_subject_data()
        
        self.aug_pipeline = DataAugmentationProcessor(methods=self.info['preprocessing']['augmentation'])
        print(f"[{self.info['dataset']['name']} Subject {self.subject_id}] Data loaded. Final shape: {self.data.shape}, Labels: {self.labels.shape}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]  # 形状: (n_bands, n_channels, window_len)
        label = self.labels[idx]
        # group_id = self.group_ids[idx]  # 如果需要，可以返回

        if self.transform:
            data = self.transform(data)
        if self.mode == "train" and self.aug_pipeline:
            data = self.aug_pipeline.process(data)


        return torch.from_numpy(data).float(), label

    def _load_and_process_subject_data(self,):
         # 1. Construct file path and load raw data
        raw_eeg_data, raw_labels = self._load_raw_data()  # raw_eeg_data: (n_sessions, n_all_channels, n_timepoints)
        raw_eeg_data = np.array(raw_eeg_data)
        raw_labels = np.array(raw_labels)
        if raw_eeg_data.shape[0] == 1:
            raw_eeg_data = raw_eeg_data[0]  # 如果第一维是1，去掉这一维
        if raw_labels.shape[0] == 1:
            raw_labels = raw_labels[0]  # 如果第一维是1，去掉这一维
        print(f"[Subject {self.subject_id}] Raw data shape: {raw_eeg_data.shape}, Raw labels shape: {raw_labels.shape}")
        
        

        # 2. Apply preprocessing pipeline

        # Resample
        resample_processor = ResampleProcessor(target_sfreq=self.target_sr, original_sfreq=self.original_sr)
        resampled_data = resample_processor.process(raw_eeg_data)
        self.sample_rate = self.target_sr

        # Artifact removal
        # artifact_removal_processor = ArtifactRemovalProcessor(method='ica', n_components=0.95)
        # cleaned_data = artifact_removal_processor.process(resampled_data)
        cleaned_data = resampled_data

        # Channel selection
        selected_eeg_data = cleaned_data[:, self.channels_selected, :]  # (n_sessions, n_selected_channels, n_timepoints)

        # Filter bank
        filter_bank_processor = FilterBankProcessor(filter_banks=self.filter_banks, sample_rate=self.sample_rate)
        processed_eeg_data = filter_bank_processor(selected_eeg_data)

        # 4. Label shift (1,2 to 0,1)
        if min(self.info['dataset']['labels']) == 1:
            adjusted_labels = raw_labels - 1
        else:
            adjusted_labels = raw_labels

        # 5. Sliding window segmentation
        self.window_len_samples = int(self.info['preprocessing']["windowing"]['window_length_sec'] * self.sample_rate)
        self.window_stride_samples = int(self.info['preprocessing']["windowing"]['window_stride_sec'] * self.sample_rate)
        windows_segmenter = SlidingWindowSegmenter(self.window_len_samples, self.window_stride_samples)
        final_data, final_labels, final_group_ids = windows_segmenter(processed_eeg_data, labels = adjusted_labels) # 基类中使用**keywords传递参数，所以需要显式说明labels参数名

        print(f"[Subject {self.subject_id}] Data loaded. Final data shape: {final_data.shape}, Label shape: {final_labels.shape}")
        return final_data, final_labels, final_group_ids

    def _load_raw_data(self, file_path):
        # This method should be implemented in the subclass to load raw data from the given file path
        raise NotImplementedError("Subclasses must implement _load_raw_data method")
    
    def _exponential_moving_standardize(self, data, init_block_size, factor_new=0.001):
        """指数移动标准化（修正版）"""
        n_ch = len(self.channels_selected)
        # 确保数据形状是 (n_channels, n_times)
        if data.shape[0] != n_ch:
            # 如果输入是 (n_times, n_channels)，则转置
            if data.shape[1] == n_ch:
                data = data.T
            else:
                # 尝试自动推断
                if data.shape[0] > data.shape[1]:
                    data = data.T
        
        df = pd.DataFrame(data.T)
        meaned = df.ewm(alpha=factor_new).mean()
        demeaned = df - meaned
        squared = demeaned * demeaned
        square_ewmed = squared.ewm(alpha=factor_new).mean()
        standardized = demeaned / np.maximum(1e-4, np.sqrt(np.array(square_ewmed)))
        standardized = np.array(standardized)  # 形状: (n_times, n_channels)
        
        if init_block_size is not None:
            # 计算初始块的标准化
            i_time_axis = 1  # 原始数据中时间轴是第1维
            # 从原始数据中取初始块
            init_data = data[..., :init_block_size]
            # 沿着时间轴计算均值和标准差
            init_mean = np.mean(init_data, axis=i_time_axis, keepdims=True)
            init_std = np.std(init_data, axis=i_time_axis, keepdims=True)
            init_block_standardized = (init_data - init_mean) / np.maximum(1e-4, init_std)
            # 注意: init_block_standardized 形状是 (n_channels, init_block_size)
            # 而 standardized[:init_block_size, :] 需要形状 (init_block_size, n_channels)
            # 所以需要转置
            standardized[:init_block_size, :] = init_block_standardized.T
        
        # 返回转置后的结果，使形状变回 (n_channels, n_times)
        return standardized.T
