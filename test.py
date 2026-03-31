import mne
import os
import numpy as np
import pandas as pd

# 1. 定义数据集根路径和被试信息
dataset_root = "src\\datasets\\27130299\\PEEG"  # 请修改为您的实际路径
subject_id = '01'  # 要读取的被试ID，例如 '01'
paradigm = 'pre'   # 实验范式，可选: 'Pre', 'IES', 'SES', 'Post', 'Follow'
run_num = '1'      # 运行编号，通常为 '1'

# 2. 根据命名规则构建文件路径
# 以读取预处理数据为例
file_name = f'sub-{subject_id}_{paradigm}_run-{run_num}_eeg.set'
file_path = os.path.join(dataset_root, f'sub-{subject_id}',"ses-1","eeg", file_name)

# 3. 检查文件是否存在
if not os.path.exists(file_path):
    # 如果预处理数据路径不存在，尝试在 derivatives 或 sourcedata 下寻找
    alt_paths = [
        os.path.join(dataset_root, 'derivatives', f'sub-{subject_id}', file_name),
        os.path.join(dataset_root, 'sourcedata', f'sub-{subject_id}', file_name.replace('_eeg.set', '_ori.set')), # 原始数据
        os.path.join(dataset_root, f'sub-{subject_id}', file_name.replace('_eeg.set', '_ori.set')) # 原始数据另一种可能位置
    ]
    for path in alt_paths:
        if os.path.exists(path):
            file_path = path
            print(f"在备用路径找到文件: {file_path}")
            break
    else:
        raise FileNotFoundError(f"未在以下路径找到数据文件: {file_path} 及其备用路径")

# 4. 使用MNE读取 .set 文件
# 注意：MNE可能需要读取同名的 .fdt 文件，请确保它与 .set 文件在同一目录
raw = mne.io.read_raw_eeglab(file_path, preload=True)
# 如果读取原始连续数据，可以使用上面的方法。
# 如果数据已经是分好段的（单个trial），可能会被读为epochs。但根据文档描述，预处理数据是分段的，但存储为连续形式+事件标记。
# 我们可以根据事件标记将数据分割成epochs。

# 5. 获取信息
print("="*50)
print(f"文件: {os.path.basename(file_path)}")
print(f"采样频率: {raw.info['sfreq']} Hz")
print(f"通道名称: {raw.info['ch_names'][:5]}...")  # 打印前5个通道名
print(f"数据形状 (通道数 x 时间点数): {raw.get_data().shape}")
print(f"事件数量: {len(raw.annotations) if raw.annotations is not None else 'N/A'}")
print("="*50)

# 6. 提取事件和epochs（如果数据是带有事件标记的连续记录）
# 文档中提到数据由事件标签（1-12）标记不同阶段（准备、提示、MI/空闲等）。
if raw.annotations is not None:
    events, event_id = mne.events_from_annotations(raw)
    print(f"找到 {len(events)} 个事件")
    print(f"事件ID与描述: {event_id}")
    
    # 例如，如果我们想提取所有“运动想象（MI）任务”阶段的数据（根据文档，标签可能为3,4,5,6）
    # 注意：实际的事件ID（整数）与描述（字符串）的映射需要查看数据或文档确定。
    # 这里仅为示例，您需要根据实际 event_id 字典调整 tmin, tmax 和 event_id 的选择。
    # 假设 '3' 对应 MI 阶段开始
    if '3' in event_id:
        epochs = mne.Epochs(raw, events, event_id={'MI_phase': event_id['3']}, 
                            tmin=0, tmax=5, baseline=None, preload=True) # 示例：提取5秒
        print(f"Epochs 数据形状: {epochs.get_data().shape}") # (epochs数, 通道数, 时间点数)
else:
    # 如果数据已经是分段的，可能没有annotations，而是每个文件就是一个trial/epoch。
    # 这种情况下，raw.get_data() 就是该trial的数据。
    data = raw.get_data()
    print(f"直接获取数据形状: {data.shape}")
    # 通常，一个 .set 文件可能包含多个trials，具体结构需参考数据集说明。
    # 根据文档，预处理数据是分好段的，'EEG.data' 是 (40, 50000)。
    # 这意味着每个文件可能是一个长段数据（可能包含多个trials），需要根据EEG.event分割。
    # 但由于我们通过MNE读取，EEG.event信息被转换成了raw.annotations。

# 7. 访问被试元数据（可选）
# participants.tsv 文件包含了所有被试的临床和人口统计学信息（如表2）
participants_tsv_path = os.path.join(dataset_root, 'participants.tsv')
if os.path.exists(participants_tsv_path):
    df_participants = pd.read_csv(participants_tsv_path, sep='\t')
    subject_info = df_participants[df_participants['participant_id'] == f'sub-{subject_id}']
    print("\n被试信息:")
    print(subject_info.to_string(index=False))
else:
    print(f"\n未在根目录找到 participants.tsv 文件。")

# 8. 数据访问示例
# 获取第一个通道的前1000个时间点数据（如果需要）
if raw.__class__.__name__ != 'Epochs':
    data_array = raw.get_data()
    # 假设我们想查看通道Cz（可能在索引'Cz'处）的数据片段
    if 'Cz' in raw.ch_names:
        cz_index = raw.ch_names.index('Cz')
        cz_data_snippet = data_array[cz_index, :1000]  # 前1000个样本
        print(f"\n通道 Cz 前1000个时间点的数据形状: {cz_data_snippet.shape}")