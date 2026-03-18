from glob import glob
from tqdm import tqdm
import mne
import torch
import scipy
import numpy as np
from typing import List, Union, Dict, Optional, Tuple
from torch.utils.data import Dataset
import yaml
import pandas as pd
import os

# 假设您有一个与XWStroke中类似的bandpass模块
# from Preprocessing import bandpass
# 为保持代码独立，这里将实现一个简单的滤波器函数
from scipy.signal import butter, filtfilt
import config  # 假设您的config模块包含必要的全局配置

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """简单的巴特沃斯带通滤波器（零相位）"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data, axis=-1)  # 假设时间在最后一个轴
    return y

class BCICompet2aIV(Dataset):
    """
    重构后的BCI Competition IV 2a数据集类。
    设计目标：提供与XWStrokeDataset完全一致的接口和输出格式。
    """
    def __init__(self, 
                 subject_id: int, 
                 dataset_info_path: str, 
                 transform = None):
        """
        初始化数据集，参数与XWStrokeDataset对齐。

        Args:
            subject_id: 受试者ID (1-9 for BCI IV 2a)。
            dataset_info_path: 数据集配置文件（YAML）路径，用于统一管理参数。
            is_test: 是否为测试模式（加载评估数据）。
            transform: 可选的额外数据增强变换。
        """
        self.subject_id = subject_id
        self.is_test = config.is_test
        self.transform = transform

        # 1. 从配置文件加载数据集元信息（统一配置入口）
        with open(dataset_info_path, 'r') as f:
            self.info = yaml.safe_load(f)
        
        # 从配置中读取BCI数据集特定的路径和参数
        self.base_path = self.info['data_dir']
        self.downsampling = config.need_resample
        self.channels_selected = self.info['channels_selected'] if 'channels_selected' in self.info else None  # 可指定通道，默认全选
        self.sample_rate = self.info["sample_rate"]  # 原始采样率
        self.target_subject = subject_id - 1  # BCI代码中subject是0-indexed
        
        # 2. 读取与XWStroke统一的预处理和切片参数
        self.window_len_samples = int(config.window_length * self.sample_rate)
        self.window_stride_samples = int(config.window_stride * self.sample_rate)
        self.filter_banks = config.filter_banks  # 例如 [[4,8], [8,12], ...]
        self.n_bands = len(self.filter_banks)

        # 3. 加载并处理数据
        # 注意：BCI IV 2a的标签是4类: 0-左手, 1-右手, 2-脚, 3-舌头
        self.data, self.labels, self.group_ids = self._load_and_process_subject_data()
        
        print(f"[BCI-IV 2a Subject {self.subject_id}] Data loaded. Final shape: {self.data.shape}, Labels: {self.labels.shape}")

    def _load_and_process_subject_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """主流程：加载、预处理、并组织单个受试者的数据，输出格式与XWStroke一致。"""
        # 1. 加载原始数据和标签
        raw_data_per_session, raw_labels = self._load_bci_raw_data()
        # raw_data_per_session: list of (n_trials, n_channels, n_times)
        # raw_labels: (n_total_trials,)
        
        # 2. 通道选择（如果配置中指定了）
        if self.channels_selected is not None:
            selected_data = [session[:, self.channels_selected, :] for session in raw_data_per_session]
        else:
            selected_data = raw_data_per_session
        
        # 3. 应用滤波器组预处理
        # 将试次列表合并为一个数组以便处理: (n_total_trials, n_channels, n_times)
        all_trials = np.concatenate(selected_data, axis=0)
        processed_data = self._apply_filterbank_preprocessing(all_trials)  # 输出: (n_trials, n_bands, n_channels, n_times)
        
        # 4. 滑动窗口分割（为每个试次生成窗口）
        # 在BCI数据中，通常每个试次已经是一个完整的 trials，但为了与XW保持一致，我们仍可进行窗口化
        # 这里将每个试次视为一个独立的“session”用于分组
        final_data, final_labels, final_group_ids = self._apply_sliding_window_to_trials(processed_data, raw_labels)
        
        return final_data, final_labels, final_group_ids

    def _load_bci_raw_data(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        简化版本：直接加载目标受试者的文件
        """
        # 文件命名规则：T为训练，E为评估
        file_suffix = 'T' if not self.is_test else 'E'
        
        # 直接构建目标文件名
        target_gdf_file = f'{self.base_path}/A{self.subject_id:02d}{file_suffix}.gdf'
        target_mat_file = f'{self.base_path}/A{self.subject_id:02d}{file_suffix}.mat'
        
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
        
        if self.downsampling != 0:
            epochs = epochs.resample(self.downsampling)
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
        
        # 原始代码中使用DataFrame，需要将数据转置为 (n_times, n_channels)
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
                filtered_data[t] = butter_bandpass_filter(
                    eeg_data[t], 
                    lowcut=l_freq, 
                    highcut=h_freq, 
                    fs=self.sample_rate
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