"""
Test script for the 5F (Five Finger) dataset loader.

Usage:
    conda activate BCI310
    python scripts/test_5f_dataset.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core import Config
from data import DATASETS


def test_5f_dataset():
    """Test loading 5F dataset for all subjects."""
    
    # Load configuration
    config = Config.fromfile('configs/dataset/5F.yaml')
    dataset_info = config.to_dict()
    
    print("=" * 70)
    print("5F Dataset Loading Test")
    print("=" * 70)
    print(f"Configuration: {dataset_info.get('dataset', {}).get('name')}")
    print(f"Data dir: {dataset_info.get('dataset', {}).get('data_dir')}")
    print(f"Target SR: {dataset_info.get('preprocessing', {}).get('resample', {}).get('target_sr')}")
    print(f"Epoch duration: {dataset_info.get('dataset', {}).get('epoch_duration_sec')} s")
    print(f"Window length: {dataset_info.get('preprocessing', {}).get('windowing', {}).get('window_length_sec')} s")
    print()
    
    # Get dataset class
    dataset_cls = DATASETS.get('5F')
    print(f"Dataset class: {dataset_cls.__name__}")
    print()
    
    subjects = dataset_info.get('dataset', {}).get('subjects', [])
    
    total_trials = 0
    all_results = []
    
    for subject_id in subjects:
        print(f"-" * 70)
        try:
            dataset = dataset_cls(subject_id=subject_id, dataset_info=dataset_info)
            info = dataset.get_info()
            
            print(f"Subject {subject_id}: OK")
            print(f"  Samples: {info['n_samples']}")
            print(f"  Data shape: {info['data_shape']}")
            print(f"  Classes: {info['n_classes']}")
            print(f"  Sample rate: {info['sample_rate']} Hz")
            
            # Verify data ranges and label distribution
            labels = dataset.labels
            unique, counts = np.unique(labels, return_counts=True)
            print(f"  Label distribution: {dict(zip(unique.tolist(), counts.tolist()))}")
            
            # Check for NaN/Inf
            data = dataset.data
            has_nan = np.isnan(data).any()
            has_inf = np.isinf(data).any()
            print(f"  Has NaN: {has_nan}, Has Inf: {has_inf}")
            print(f"  Data range: [{data.min():.2f}, {data.max():.2f}]")
            
            total_trials += info['n_samples']
            all_results.append({
                'subject_id': subject_id,
                'status': 'OK',
                'n_samples': info['n_samples'],
                'data_shape': info['data_shape'],
                'n_classes': info['n_classes']
            })
            
        except Exception as e:
            print(f"Subject {subject_id}: FAILED - {e}")
            all_results.append({
                'subject_id': subject_id,
                'status': 'FAILED',
                'error': str(e)
            })
    
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total subjects tested: {len(subjects)}")
    print(f"Successful: {sum(1 for r in all_results if r['status'] == 'OK')}")
    print(f"Failed: {sum(1 for r in all_results if r['status'] == 'FAILED')}")
    print(f"Total samples (before sliding window): {total_trials}")
    
    # Also test a single sample retrieval
    print()
    print("-" * 70)
    print("Testing __getitem__ for Subject 1...")
    dataset = dataset_cls(subject_id=1, dataset_info=dataset_info)
    if len(dataset) > 0:
        sample, label = dataset[0]
        print(f"  Sample type: {type(sample).__name__}")
        print(f"  Sample shape: {tuple(sample.shape)}")
        print(f"  Label: {label.item()}")
        print(f"  Sample dtype: {sample.dtype}")
        print(f"  Label dtype: {label.dtype}")
    
    print()
    print("All tests completed!")


if __name__ == '__main__':
    test_5f_dataset()
