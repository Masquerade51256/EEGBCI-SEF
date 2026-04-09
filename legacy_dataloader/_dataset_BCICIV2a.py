import mne
import scipy
import os
import numpy as np

# from preprocessing.pipline import FilterBankProcessor, ExponentialMovingStandardizeProcessor
from legacy_dataloader._dataset_Base import BaseDataset


class BCICompet2aIV(BaseDataset):
    def _load_raw_data(self):

        # 文件命名规则：T为训练，E为评估
        file_suffix = 'T' if self.mode == "train" else 'E'
        
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
        if self.mode == "train":
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

        print("Epochs data shape:", epochs_data.shape)
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
