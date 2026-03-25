import mne
import scipy
import numpy as np
from typing import List, Union, Dict, Optional, Tuple
import os

from preprocessing.bandpass import bandpass_filter
# from preprocessing.pipline import FilterBankProcessor, ExponentialMovingStandardizeProcessor
from dataloader._dataset_Base import BaseDataset


class BCICompet2aIV(BaseDataset):

    def _load_raw_data(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        简化版本：直接加载目标受试者的文件
        """
        # 文件命名规则：T为训练，E为评估
        file_suffix = 'T' if not self.is_test else 'E'
        
        # 直接构建目标文件名
        target_gdf_file = f'{self.data_dir}/A{self.subject_id:02d}{file_suffix}.gdf'
        target_mat_file = f'{self.data_dir}/A{self.subject_id:02d}{file_suffix}.mat'
        
        # 检查文件是否存在
        if not os.path.exists(target_gdf_file):
            raise FileNotFoundError(f"GDF file not found: {target_gdf_file}")
        if not os.path.exists(target_mat_file):
            raise FileNotFoundError(f"Label file not found: {target_mat_file}")
        
        print(f'Loading BCI IV 2a Subject {self.subject_id} ({file_suffix} file)')
        
        # 加载数据和标签
        session_data, label_list = self._load_single_session(target_gdf_file, target_mat_file)
        
        # 返回格式与XWStrokeDataset一致
        return [session_data], label_list

    def _load_single_session(self, gdf_path, mat_path):
        """加载单个session的数据"""
        # 使用MNE读取GDF文件
        raw = mne.io.read_raw_gdf(gdf_path, preload=True)
        events, annot = mne.events_from_annotations(raw)
        
        raw.load_data()
        raw.filter(0., 40., fir_design='firwin')
        raw.info['bads'] += ['EOG-left', 'EOG-central', 'EOG-right']
        
        picks = mne.pick_types(raw.info,
                            meg=False,
                            eeg=True,
                            eog=False,
                            stim=False,
                            exclude='bads')
        
        # 定义事件ID
        tmin, tmax = 0, 3
        if not self.is_test:
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
                            preload=True)
        
        self.fs = epochs.info['sfreq']
        
        # 获取数据
        epochs_data = epochs.get_data() * 1e6
        
        # 应用标准化
        splited_data = []
        for epoch in epochs_data:
            normalized_data = self._exponential_moving_standardize(
                epoch,
                init_block_size=int(raw.info['sfreq'] * 4)
            )
            splited_data.append(normalized_data)
        session_data = np.stack(splited_data)
        
        # 加载标签
        label_list = scipy.io.loadmat(mat_path)['classlabel'].reshape(-1) - 1
        
        return session_data, label_list
    
    def _exponential_moving_standardize(self, data, init_block_size, factor_new=0.001):
        """指数移动标准化（修正版）"""
        import pandas as pd
        import numpy as np
        
        # 确保数据形状是 (n_channels, n_times)
        if data.shape[0] != 22:  # 假设22是通道数
            # 如果输入是 (n_times, n_channels)，则转置
            if data.shape[1] == 22:
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

    def _apply_filterbank_preprocessing(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        应用滤波器组预处理，与XWStrokeDataset中的逻辑一致。
        输入: (n_trials, n_channels, n_times)
        输出: (n_trials, n_bands, n_channels, n_times)
        """
        n_trials, n_channels, n_times = eeg_data.shape
        band_data_list = []
        
        for band_idx, (l_freq, h_freq) in enumerate(self.filter_banks):
            # 对每个试次和通道应用带通滤波
            filtered_data = np.zeros_like(eeg_data)
            for t in range(n_trials):
                # 这里使用简单的巴特沃斯滤波器，您可以用自己的bandpass模块替换
                filtered_data[t] = bandpass_filter(
                    eeg_data[t], 
                    low_freq=l_freq, 
                    high_freq=h_freq, 
                    sfreq=self.sample_rate
                )
            
            # 通道标准化（沿时间轴）
            mean = filtered_data.mean(axis=-1, keepdims=True)
            std = filtered_data.std(axis=-1, keepdims=True)
            std[std < 1e-8] = 1.0
            normalized_band_data = (filtered_data - mean) / std
            
            # 添加频带维度
            normalized_band_data = normalized_band_data[:, np.newaxis, :, :]  # (n_trials, 1, n_channels, n_times)
            band_data_list.append(normalized_band_data)
        
        # 沿频带维度拼接
        multi_band_data = np.concatenate(band_data_list, axis=1)  # (n_trials, n_bands, n_channels, n_times)
        return multi_band_data

    def _apply_sliding_window_to_trials(self, processed_data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        对每个试次的数据应用滑动窗口，为每个窗口分配组ID（试次ID）。
        输出格式与XWStrokeDataset._apply_sliding_window完全一致。
        """
        n_trials, n_bands, n_channels, n_times = processed_data.shape
        window_len = self.window_len_samples
        window_stride = self.window_stride_samples
        
        all_windows = []
        all_labels = []
        all_group_ids = []
        
        for trial_idx in range(n_trials):
            trial_data = processed_data[trial_idx]  # (n_bands, n_channels, n_times)
            trial_label = labels[trial_idx]
            
            # 计算该试次可以产生的窗口数
            n_windows = 1 + (n_times - window_len) // window_stride
            
            for win_idx in range(n_windows):
                start = win_idx * self.window_stride_samples
                end = start + self.window_len_samples
                window = trial_data[:, :, start:end]  # (n_bands, n_channels, window_len)
                
                all_windows.append(window)
                all_labels.append(trial_label)
            
            # 该试次生成的所有窗口共享同一个组ID
            all_group_ids.extend([trial_idx] * n_windows)
        
        if all_windows:
            all_windowed_data = np.stack(all_windows, axis=0).astype(np.float32)  # (n_total_windows, n_bands, n_channels, window_len)
            all_windowed_labels = np.array(all_labels, dtype=np.int64)
            all_group_ids = np.array(all_group_ids, dtype=np.int64)
        else:
            # 如果没有生成窗口（例如，时间长度小于窗口长度），则返回空数组
            all_windowed_data = np.array([], dtype=np.float32).reshape(0, n_bands, n_channels, window_len)
            all_windowed_labels = np.array([], dtype=np.int64)
            all_group_ids = np.array([], dtype=np.int64)
        
        return all_windowed_data, all_windowed_labels, all_group_ids


    def __getitem__(self, idx):
        data = self.data[idx]  # 形状: (n_bands, n_channels, window_len)
        label = self.labels[idx]
        # group_id = self.group_ids[idx]  # 如果需要，可以返回

        if self.transform:
            data = self.transform(data)

        # 返回格式与XWStrokeDataset一致
        return data, label