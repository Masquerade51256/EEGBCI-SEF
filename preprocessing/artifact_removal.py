# preprocessing/pipeline.py
import numpy as np
from scipy import signal
import mne
from preprocessing.base_processor import BaseProcessor
# ... 其他导入和已有的 FilterBankProcessor 等 ...

class ArtifactRemovalProcessor(BaseProcessor):
    """伪影去除处理器。提供多种方法可选。"""
    def __init__(self, method: str = 'ica', **kwargs):
        """
        Args:
            method: 伪影去除方法，可选 'ica'（独立成分分析）, 'regression', 'ads'（自适应滤波）等。
            kwargs: 对应方法的参数，如 n_components for ICA。
        """
        self.method = method
        self.params = kwargs
        
    def process(self, data: np.ndarray, info=None) -> np.ndarray:
        """
        Args:
            data: EEG数据，形状 (n_channels, n_times) 或 (n_trials, n_channels, n_times)。
            info: 可选的mne.Info对象，包含通道位置等信息，某些方法（如ICA）需要。
        Returns:
            cleaned_data: 去除伪影后的数据。
        """
        if self.method == 'ica':
            return self._remove_by_ica(data, info)
        elif self.method == 'regression':
            return self._remove_by_regression(data)
        # ... 其他方法 ...
        else:
            raise ValueError(f"Unsupported artifact removal method: {self.method}")

    def _remove_by_ica(self, data, info):
        """使用MNE库进行ICA去伪影。"""
        # 注意：ICA通常需要在连续数据或长片段上进行，对计算资源要求较高。
        # 这里仅为示例框架，实际实现需考虑数据格式和性能。
        if info is None:
            # 创建一个简易的info对象，如果未提供
            info = mne.create_info(ch_names=[f'CH{i}' for i in range(data.shape[-2])], sfreq=250, ch_types='eeg')
        raw = mne.io.RawArray(data, info)
        ica = mne.preprocessing.ICA(n_components=self.params.get('n_components', 0.95), random_state=97)
        ica.fit(raw)
        # 这里需要定义或自动检测伪影成分。实际操作更复杂，可能需EOG/ECG通道或模板匹配。
        # 例如: ica.exclude = [0, 1] # 假设前两个成分为伪影
        ica.apply(raw)
        return raw.get_data()
    
    def _remove_by_regression(self, data):
        """使用回归方法去除伪影（例如，利用EOG参考通道）。"""
        # 简化示例：假设最后两个通道是EOG参考通道
        eeg_data = data[..., :-2, :]
        eog_data = data[..., -2:, :]
        # 为每个EEG通道对EOG通道进行回归并减去
        # ... 实现回归计算 ...
        return cleaned_eeg

class DataAugmentationProcessor(BaseProcessor):
    """数据增强处理器。在样本级别（trial/window）进行操作。"""
    def __init__(self, methods: list, p: float = 0.5):
        """
        Args:
            methods: 增强方法列表，如 ['time_warp', 'add_noise', 'channel_dropout']。
            p: 每个样本应用增强的概率。
        """
        self.methods = methods
        self.p = p
        
    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Args:
            data: 单个样本数据，形状 (n_bands, n_channels, n_times) 或 (n_channels, n_times)。
        Returns:
            augmented_data: 增强后的数据。
        """
        if np.random.random() > self.p:
            return data
            
        aug_data = data.copy()
        for method in self.methods:
            if method == 'time_warp':
                aug_data = self._time_warp(aug_data)
            elif method == 'add_noise':
                aug_data = self._add_gaussian_noise(aug_data)
            elif method == 'channel_dropout':
                aug_data = self._channel_dropout(aug_data)
            # ... 其他增强方法 ...
        return aug_data
    
    def _time_warp(self, data, sigma=0.2):
        """轻微的时间扭曲。"""
        from scipy.interpolate import CubicSpline
        n_time = data.shape[-1]
        orig_time = np.arange(n_time)
        warp_steps = np.random.normal(loc=1.0, scale=sigma, size=n_time).cumsum()
        warp_steps = (warp_steps - warp_steps.min()) / (warp_steps.max() - warp_steps.min()) * (n_time - 1)
        warped_data = np.apply_along_axis(lambda x: CubicSpline(orig_time, x)(warp_steps), -1, data)
        return warped_data
    
    def _add_gaussian_noise(self, data, snr_db=20):
        """添加高斯噪声，信噪比为snr_db。"""
        signal_power = np.mean(data**2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), data.shape)
        return data + noise
    
    def _channel_dropout(self, data, dropout_rate=0.1):
        """随机将部分通道置零，模拟坏导或注意力转移。"""
        n_channels = data.shape[-2]
        n_to_drop = int(n_channels * dropout_rate)
        if n_to_drop > 0:
            channels_to_drop = np.random.choice(n_channels, n_to_drop, replace=False)
            # 假设通道维度是倒数第二维
            slc = [slice(None)] * data.ndim
            slc[-2] = channels_to_drop
            data[tuple(slc)] = 0
        return data