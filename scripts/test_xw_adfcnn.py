#!/usr/bin/env python3
"""Test script for XWStroke + ADFCNN experiment"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import Config, ExperimentManager
from models import register_all_models
from data import DATASETS
from trainers import TRAINERS

# Register models
register_all_models()

def main():
    print("=" * 60)
    print("Testing XWStroke + ADFCNN Experiment")
    print("=" * 60)
    
    # Load config
    config = Config.fromfile('configs/experiment/xw_adfcnn.yaml')
    
    # Modify config for quick test (1 subject, few epochs)
    config.set('data.subjects', [1])
    config.set('training.epochs', 2)
    config.set('experiment.name', 'xw_adfcnn_quick_test')
    
    print("\nConfiguration:")
    print(f"  Dataset: {config.data.dataset}")
    print(f"  Model: {config.model.type}")
    print(f"  Subjects: {config.data.subjects}")
    print(f"  Epochs: {config.training.epochs}")
    
    # Create experiment
    try:
        exp = ExperimentManager(config, exp_name='xw_adfcnn_test')
        exp.setup()
        print("\n[OK] Experiment setup successful")
        
        # Build dataset
        exp.build_dataset()
        print(f"[OK] Dataset built: {len(exp.datasets['instances'])} subjects")
        
        # Check data
        for subj_id, dataset in exp.datasets['instances'].items():
            print(f"  Subject {subj_id}: data shape {dataset.data.shape}, labels {dataset.labels.shape}")
        
        # Build model
        exp.build_model()
        print(f"[OK] Model built: {type(exp.model).__name__}")
        
        # Build trainer
        exp.build_trainer()
        print(f"[OK] Trainer built: {type(exp.trainer).__name__}")
        
        # Run training (just 1 fold for quick test)
        print("\nRunning training...")
        results = exp.run()
        
        print("\n" + "=" * 60)
        print("Test completed successfully!")
        print(f"Results: {results}")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
