from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import create_windows_from_events
from Models.EEGNet import EEGNet
from braindecode.classifier import EEGClassifier
import torch

# 1. 加载数据并生成时间窗口
dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[1])
windows = create_windows_from_events(dataset, trial_start_offset_samples=0, trial_stop_offset_samples=0)

# 2. 初始化模型
# model = EEGNetv4(n_channels=22, n_times=1125, n_classes=4)
model = EEGNet(num_channels=22)

# 3. 训练分类器
clf = EEGClassifier(model, optimizer=torch.optim.Adam, iterator_train__shuffle=True)
clf.fit(windows, y=None)  # y 自动从窗口元数据中提取

# 4. 预测
y_pred = clf.predict(windows)