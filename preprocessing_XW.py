"""
从XWStroke数据集中提取运动想象阶段4秒EEG数据
将每个被试的40个试次中，运动想象阶段的4秒数据提取出来
输出形状：(40, 32, 2000)  # 40试次 × 32通道 × 2000时间点(4秒×500Hz)
"""

import scipy.io
import numpy as np
import os
import warnings
from pathlib import Path

def extract_motor_imagery_4s(input_file, output_file=None, event_marker=2, fs=500):
    """
    从单个被试的MAT文件中提取运动想象阶段的4秒EEG数据
    
    参数:
    ----------
    input_file : str
        输入的MAT文件路径
    output_file : str, 可选
        输出文件路径，如为None则自动生成
    event_marker : int, 默认=1
        事件标记通道中标识"运动想象开始"的标记值
    fs : int, 默认=500
        采样率(Hz)
    
    返回:
    ----------
    extracted_data : numpy.ndarray
        提取的EEG数据，形状为(n_trials, 32, 2000)
    labels : numpy.ndarray
        试次标签，形状为(n_trials,)
    """
    
    # 1. 加载MAT文件
    try:
        data = scipy.io.loadmat(input_file)
        # 找到包含EEG数据的变量
        for key, value in data.items():
            if not key.startswith('__'):
                eeg_data = value[0][0][0]  # 形状应为: (n_sessions, n_channels, n_timepoints)
                labels = value[0][0][1]    # 形状应为: (n_sessions, 1)
                break
    except Exception as e:
        raise FileNotFoundError(f"无法加载文件 {input_file}: {e}")
    
    # 2. 检查数据形状
    n_trials, n_channels, n_timepoints = eeg_data.shape
    print(f"原始数据形状: {eeg_data.shape}")
    print(f"标签形状: {labels.shape}")
    
    # 确保是40个试次
    if n_trials != 40:
        warnings.warn(f"试次数不是40，而是{n_trials}，但将继续处理")
    
    # 确保是33个通道（32个EEG + 1个事件标记）
    if n_channels != 33:
        raise ValueError(f"期望33个通道(32EEG+1事件标记)，但实际有{n_channels}个通道")
    
    # 3. 提取4秒运动想象数据
    extracted_trials = []
    valid_trials = 0
    
    # 计算4秒对应的时间点数
    duration_4s = 4 * fs  # 4秒 × 500Hz = 2000个点
    
    for trial_idx in range(n_trials):
        # 获取当前试次的所有通道数据
        trial_data = eeg_data[trial_idx, :, :]  # 形状: (33, n_timepoints)
        
        # 分离EEG通道和事件标记通道
        eeg_channels = trial_data[:32, :]      # 前32个是EEG通道
        event_channel = trial_data[32, :]     # 第33个是事件标记通道
        
        # 在事件标记通道中寻找运动想象开始的标记
        # 找到事件标记值等于event_marker的位置
        event_indices = np.where(event_channel == event_marker)[0]
        # print(event_channel)
        print(f"试次{trial_idx+1}: 事件标记值{event_marker}出现的索引: {event_indices}")
        if len(event_indices) == 0:
            # 如果没找到指定的标记值，尝试寻找第一个非零值
            event_indices = np.where(event_channel != 0)[0]
            if len(event_indices) > 0:
                warnings.warn(f"试次{trial_idx+1}: 未找到标记值{event_marker}，使用第一个非零事件标记")
            else:
                # 如果仍然找不到，使用默认值（假设从2秒后开始，即1000个点）
                warnings.warn(f"试次{trial_idx+1}: 未找到事件标记，使用默认起始点1000")
                event_indices = [1000]
        
        # 取第一个事件标记作为运动想象开始
        start_idx = event_indices[0]
        
        # 检查索引范围是否有效
        if start_idx + duration_4s > n_timepoints:
            warnings.warn(f"试次{trial_idx+1}: 起始索引{start_idx}加上4秒超出数据范围，跳过此试次")
            continue
        
        # 提取4秒的EEG数据（去除事件标记通道）
        extracted_eeg = eeg_channels[:, start_idx:start_idx + duration_4s]
        
        # 验证提取的形状
        if extracted_eeg.shape != (32, duration_4s):
            warnings.warn(f"试次{trial_idx+1}: 提取的数据形状为{extracted_eeg.shape}，期望(32, {duration_4s})")
            continue
        
        extracted_trials.append(extracted_eeg)
        valid_trials += 1
    
    # 4. 转换为numpy数组
    if len(extracted_trials) == 0:
        raise ValueError("未成功提取任何试次数据，请检查事件标记值或数据")
    
    extracted_data = np.array(extracted_trials)  # 形状: (n_valid_trials, 32, 2000)
    
    print(f"成功提取 {valid_trials}/{n_trials} 个试次")
    print(f"提取后数据形状: {extracted_data.shape}")
    
    # 5. 保存为MAT文件
    if output_file is None:
        # 自动生成输出文件名
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_4s{input_path.suffix}"
    
    # 创建输出字典
    output_dict = {
        'eeg_data_4s': extracted_data,  # 提取的4秒运动想象EEG
        'labels': labels.flatten(),      # 原始标签
        'fs': fs,                        # 采样率
        'duration': 4.0,                 # 持续时间(秒)
        'n_channels': 32,                # 通道数
        'event_marker_used': event_marker,  # 使用的事件标记值
        'source_file': input_file        # 源文件
    }
    
    # 保存MAT文件
    scipy.io.savemat(output_file, output_dict)
    print(f"数据已保存到: {output_file}")
    
    return extracted_data, labels.flatten()

def batch_process_all_subjects(data_dir, subject_ids=None, event_marker=1):
    """
    批量处理所有被试的数据
    
    参数:
    ----------
    data_dir : str
        数据目录的根路径
    subject_ids : list, 可选
        被试ID列表，如为None则自动查找
    event_marker : int, 默认=1
        事件标记值
    """
    
    data_dir = Path(data_dir)
    
    # 如果未指定被试ID，自动查找
    if subject_ids is None:
        # 查找所有被试文件夹
        subject_folders = [f for f in data_dir.iterdir() if f.is_dir() and f.name.startswith('sub-')]
        subject_ids = []
        for folder in subject_folders:
            try:
                # 从文件夹名提取ID，如"sub-01" -> 1
                sub_id = int(folder.name.split('-')[1])
                subject_ids.append(sub_id)
            except:
                continue
        subject_ids.sort()
    
    print(f"找到 {len(subject_ids)} 个被试: {subject_ids}")
    
    # 处理每个被试
    all_results = {}
    
    for sub_id in subject_ids:
        print(f"\n{'='*50}")
        print(f"处理被试 sub-{sub_id:02d}")
        print('='*50)
        
        try:
            # 构建输入文件路径
            sub_folder = data_dir / f"sub-{sub_id:02d}"
            input_file = sub_folder / f"sub-{sub_id:02d}_task-motor-imagery_eeg.mat"
            
            if not input_file.exists():
                print(f"警告: 文件不存在 {input_file}，跳过")
                continue
            
            # 处理单个被试
            extracted_data, labels = extract_motor_imagery_4s(
                input_file=str(input_file),
                event_marker=event_marker
            )
            
            all_results[sub_id] = {
                'data': extracted_data,
                'labels': labels,
                'status': 'success'
            }
            
        except Exception as e:
            print(f"处理被试 sub-{sub_id:02d} 时出错: {e}")
            all_results[sub_id] = {
                'data': None,
                'labels': None,
                'status': f'error: {str(e)}'
            }
    
    # 打印汇总信息
    print(f"\n{'='*50}")
    print("处理完成汇总:")
    print('='*50)
    
    success_count = sum(1 for res in all_results.values() if res['status'] == 'success')
    print(f"成功处理: {success_count}/{len(subject_ids)} 个被试")
    
    return all_results

if __name__ == "__main__":
    # 使用示例
    DATA_DIR = "src/datasets/21679035/sourcedata"  # 修改为您的数据目录路径
    
    # 方法1: 处理单个被试
    # input_file = "/path/to/sub-01/sub-01_task-motor-imagery_eeg.mat"
    # extracted_data, labels = extract_motor_imagery_4s(input_file, event_marker=1)
    
    # 方法2: 批量处理所有被试
    # 指定要处理的被试ID列表
    subject_ids_to_process = list(range(1,51))
    
    # 或者设置为None自动查找所有被试
    # subject_ids_to_process = None
    
    results = batch_process_all_subjects(
        data_dir=DATA_DIR,
        subject_ids=subject_ids_to_process,
        event_marker=2  # 事件标记值，根据实际数据调整
    )