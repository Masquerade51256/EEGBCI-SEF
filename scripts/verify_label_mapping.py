"""
验证 XWStroke 数据集的标签映射（左右手 -> 患侧/健侧）是否正确。

用法:
    conda activate BCI310
    python scripts/verify_label_mapping.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml
import numpy as np
import pandas as pd

from data.datasets import XWStrokeDataset, _load_paralysis_side


def test_paralysis_side_loading():
    """测试 participants.tsv 读取是否正确。"""
    print("=" * 60)
    print("测试 1: 读取 participants.tsv")
    print("=" * 60)
    
    participants = pd.read_csv("src/datasets/21679035/participants.tsv", sep='\t')
    
    test_subjects = [
        (1, 'right'),   # sub-01: right paralysis
        (2, 'left'),    # sub-02: left paralysis
        (6, 'left'),    # sub-06: left paralysis (且为左利手)
        (15, 'left'),   # sub-15: left paralysis (且为左利手)
    ]
    
    dataset_info = {
        'dataset': {
            'data_dir': 'src/datasets/21679035/sourcedata',
            'participants_tsv': 'src/datasets/21679035/participants.tsv',
        }
    }
    
    all_pass = True
    for sub_id, expected in test_subjects:
        side = _load_paralysis_side(sub_id, dataset_info)
        status = "[OK]" if side == expected else "[FAIL]"
        if side != expected:
            all_pass = False
        print(f"  {status} sub-{sub_id:02d}: ParalysisSide = {side} (期望: {expected})")
    
    print()
    return all_pass


def test_label_mapping():
    """测试标签映射逻辑是否正确。"""
    print("=" * 60)
    print("测试 2: 标签映射逻辑验证")
    print("=" * 60)
    
    # 加载数据集配置
    with open("configs/dataset/XWStroke_affected.yaml", 'r', encoding='utf-8') as f:
        dataset_info = yaml.safe_load(f)
    
    # 也加载原始 lr 配置用于对比
    with open("configs/dataset/XWStroke.yaml", 'r', encoding='utf-8') as f:
        dataset_info_lr = yaml.safe_load(f)
    
    # 选择有代表性的被试：右偏瘫 + 左偏瘫
    test_subjects = [
        (1, 'right'),   # sub-01: 右偏瘫 -> 左=健侧, 右=患侧 -> 应翻转
        (2, 'left'),    # sub-02: 左偏瘫 -> 左=患侧, 右=健侧 -> 不变
    ]
    
    all_pass = True
    for sub_id, paralysis_side in test_subjects:
        print(f"\n--- sub-{sub_id:02d} (ParalysisSide: {paralysis_side}) ---")
        
        # 加载原始标签 (lr 模式)
        ds_lr = XWStrokeDataset(subject_id=sub_id, dataset_info=dataset_info_lr)
        labels_lr = ds_lr.labels
        unique_lr, counts_lr = np.unique(labels_lr, return_counts=True)
        
        # 加载映射后标签 (affected 模式)
        ds_aff = XWStrokeDataset(subject_id=sub_id, dataset_info=dataset_info)
        labels_aff = ds_aff.labels
        unique_aff, counts_aff = np.unique(labels_aff, return_counts=True)
        
        print(f"  原始标签 (0=左, 1=右):     {dict(zip(unique_lr, counts_lr))}")
        print(f"  映射后标签 (0=患侧, 1=健侧): {dict(zip(unique_aff, counts_aff))}")
        
        # 验证映射逻辑
        # 原始: 0=左, 1=右
        if paralysis_side == 'left':
            # 左偏瘫: 左(0)=患侧, 右(1)=健侧 -> 不应翻转
            expected = labels_lr.copy()
        else:  # 'right'
            # 右偏瘫: 左(0)=健侧, 右(1)=患侧 -> 应翻转
            expected = 1 - labels_lr
        
        if np.array_equal(labels_aff, expected):
            print(f"  [OK] 映射结果与理论预期一致")
        else:
            print(f"  [FAIL] 映射结果与理论预期不一致!")
            print(f"    预期: {expected[:10]}...")
            print(f"    实际: {labels_aff[:10]}...")
            all_pass = False
        
        # 额外验证：两类都应该存在（对于平衡数据集）
        if len(unique_aff) == 2:
            print(f"  [OK] 两类标签均存在")
        else:
            print(f"  [FAIL] 标签类别不完整: {unique_aff}")
            all_pass = False
    
    print()
    return all_pass


def test_statistics():
    """统计所有被试的标签分布。"""
    print("=" * 60)
    print("测试 3: 全被试标签分布统计")
    print("=" * 60)
    
    with open("configs/dataset/XWStroke_affected.yaml", 'r', encoding='utf-8') as f:
        dataset_info = yaml.safe_load(f)
    
    participants = pd.read_csv("src/datasets/21679035/participants.tsv", sep='\t')
    
    affected_counts = []
    unaffected_counts = []
    
    for sub_id in range(1, 51):
        ds = XWStrokeDataset(subject_id=sub_id, dataset_info=dataset_info)
        labels = ds.labels
        unique, counts = np.unique(labels, return_counts=True)
        
        sub_info = participants[participants['Participant_ID'] == f'sub-{sub_id:02d}']
        paralysis_side = sub_info['ParalysisSide'].values[0] if len(sub_info) > 0 else 'unknown'
        handedness = sub_info['Handedness'].values[0] if len(sub_info) > 0 else 'unknown'
        
        affected_counts.append(counts[0] if 0 in unique else 0)
        unaffected_counts.append(counts[1] if 1 in unique else 0)
        
        print(f"  sub-{sub_id:02d} | 偏瘫侧: {paralysis_side:5} | 利手: {handedness:5} | "
              f"患侧(0): {affected_counts[-1]:4} | 健侧(1): {unaffected_counts[-1]:4}")
    
    print(f"\n汇总: 患侧试次总数 = {sum(affected_counts)}, 健侧试次总数 = {sum(unaffected_counts)}")
    print()


def main():
    print("\n" + "=" * 60)
    print("XWStroke 标签映射验证脚本")
    print("=" * 60 + "\n")
    
    passed = True
    passed &= test_paralysis_side_loading()
    passed &= test_label_mapping()
    test_statistics()
    
    print("=" * 60)
    if passed:
        print("[PASS] 所有测试通过！标签映射实现正确。")
    else:
        print("[FAIL] 部分测试失败，请检查实现。")
    print("=" * 60 + "\n")
    
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
