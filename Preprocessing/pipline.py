# preprocessing/pipeline.py
from abc import ABC, abstractmethod
import numpy as np

class BaseProcessor(ABC):
    """所有预处理步骤的抽象基类。"""
    @abstractmethod
    def process(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """处理数据并返回处理后的数据。"""
        pass

    def __call__(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return self.process(data, **kwargs)


# 具体处理器实现
from preprocessing.bandpass import bandpass_filter  # 假设这是您的滤波函数

class FilterBankProcessor(BaseProcessor):
    """滤波器组处理器，对应XWStrokeDataset中的_apply_preprocessing核心部分。"""
    def __init__(self, filter_banks, sample_rate):
        self.filter_banks = filter_banks
        self.sample_rate = sample_rate

    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Args:
            data: Shape (n_trials, n_channels, n_times) OR (n_channels, n_times)
        Returns:
            multi_band_data: Shape (..., n_bands, n_channels, n_times)
        """
        # 处理单一样本（无trials维度）的情况
        if data.ndim == 2:
            data = np.expand_dims(data, axis=0)
            has_trial_dim = False
        else:
            has_trial_dim = True

        n_trials, n_channels, n_times = data.shape
        band_data_list = []
        
        for l_freq, h_freq in self.filter_banks:
            filtered_data = np.zeros_like(data)
            for t in range(n_trials):
                # 调用统一的滤波函数
                filtered_data[t] = bandpass_filter(data[t], low_freq=l_freq, high_freq=h_freq, sfreq=self.sample_rate)
            # 逐通道标准化
            mean = filtered_data.mean(axis=-1, keepdims=True)
            std = filtered_data.std(axis=-1, keepdims=True)
            std[std < 1e-8] = 1.0
            normalized_band_data = (filtered_data - mean) / std
            # 增加频带维度
            normalized_band_data = normalized_band_data[:, np.newaxis, :, :]
            band_data_list.append(normalized_band_data)
        
        multi_band_data = np.concatenate(band_data_list, axis=1)  # (n_trials, n_bands, n_channels, n_times)
        
        if not has_trial_dim:
            multi_band_data = np.squeeze(multi_band_data, axis=0) # (n_bands, n_channels, n_times)
        return multi_band_data


class ExponentialMovingStandardizeProcessor(BaseProcessor):
    def __init__(self, init_block_size: int, num_channels: int, factor_new: float = 0.001):
        self.init_block_size = init_block_size
        self.num_channels = num_channels
        self.factor_new = factor_new
        # 可以在这里导入_pandas_和_numpy_
    
    def process(self, data: np.ndarray) -> np.ndarray:
        import pandas as pd
        import numpy as np
        
        # 确保数据形状是 (n_channels, n_times)
        if data.shape[0] != self.num_channels:
            # 如果输入是 (n_times, n_channels)，则转置
            if data.shape[1] == self.num_channels:
                data = data.T
            else:
                # 尝试自动推断
                if data.shape[0] > data.shape[1]:
                    data = data.T
        
        df = pd.DataFrame(data.T)
        meaned = df.ewm(alpha=self.factor_new).mean()
        demeaned = df - meaned
        squared = demeaned * demeaned
        square_ewmed = squared.ewm(alpha=self.factor_new).mean()
        standardized = demeaned / np.maximum(1e-4, np.sqrt(np.array(square_ewmed)))
        standardized = np.array(standardized)  # 形状: (n_times, n_channels)
        
        if self.init_block_size is not None:
            # 计算初始块的标准化
            i_time_axis = 1  # 原始数据中时间轴是第1维
            # 从原始数据中取初始块
            init_data = data[..., :self.init_block_size]
            # 沿着时间轴计算均值和标准差
            init_mean = np.mean(init_data, axis=i_time_axis, keepdims=True)
            init_std = np.std(init_data, axis=i_time_axis, keepdims=True)
            init_block_standardized = (init_data - init_mean) / np.maximum(1e-4, init_std)
            # 注意: init_block_standardized 形状是 (n_channels, init_block_size)
            # 而 standardized[:init_block_size, :] 需要形状 (init_block_size, n_channels)
            # 所以需要转置
            standardized[:self.init_block_size, :] = init_block_standardized.T
        
        # 返回转置后的结果，使形状变回 (n_channels, n_times)
        return standardized.T


class ChannelSelector(BaseProcessor):
    """通道选择处理器。"""
    def __init__(self, channels_to_select):
        self.channels_to_select = channels_to_select

    def process(self, data: np.ndarray) -> np.ndarray:
        return data[..., self.channels_to_select, :]