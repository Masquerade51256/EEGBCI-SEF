#!/usr/bin/env python3
"""
Test script to verify the new framework is set up correctly.

This script tests:
1. Component registration (models, datasets, trainers)
2. Configuration loading
3. Basic imports
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from core import Config, ExperimentManager, DATASETS, MODELS, TRAINERS
        print("  [OK] Core modules imported")
    except ImportError as e:
        print(f"  [FAIL] Core modules import failed: {e}")
        return False
    
    try:
        from core.base import BaseTrainer, BaseDataset
        print("  [OK] Base classes imported")
    except ImportError as e:
        print(f"  [FAIL] Base classes import failed: {e}")
        return False
    
    try:
        from utils import PathManager, setup_logger
        print("  [OK] Utils imported")
    except ImportError as e:
        print(f"  [FAIL] Utils import failed: {e}")
        return False
    
    return True


def test_registrations():
    """Test that components are registered."""
    print("\nTesting component registrations...")
    
    from models import register_all_models
    register_all_models()
    
    # Import data and trainers to trigger registration
    from data import DATASETS
    from trainers import TRAINERS
    from core import MODELS
    
    # Check models
    models = MODELS.list_keys()
    print(f"  Registered models ({len(models)}): {models}")
    
    # Check datasets
    datasets = DATASETS.list_keys()
    print(f"  Registered datasets ({len(datasets)}): {datasets}")
    
    # Check trainers
    trainers = TRAINERS.list_keys()
    print(f"  Registered trainers ({len(trainers)}): {trainers}")
    
    return len(models) > 0 and len(datasets) > 0 and len(trainers) > 0


def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    from core import Config
    
    # Test config creation
    config = Config({
        'experiment': {'name': 'test'},
        'data': {'dataset': 'XWStroke'},
        'model': {'type': 'EEGNet'}
    })
    
    print(f"  Config created: {config.experiment.name}")
    print(f"  Dataset: {config.data.dataset}")
    print(f"  Model: {config.model.type}")
    
    # Test config loading
    config_path = 'configs/experiment/default.yaml'
    if os.path.exists(config_path):
        try:
            loaded_config = Config.fromfile(config_path)
            print(f"  [OK] Config loaded from {config_path}")
        except Exception as e:
            print(f"  [FAIL] Config loading failed: {e}")
            return False
    else:
        print(f"  [WARN] Config file not found: {config_path}")
    
    return True


def test_model_creation():
    """Test model instantiation."""
    print("\nTesting model creation...")
    
    try:
        import torch
        from core.registry import build_from_config, MODELS
        from models import register_all_models
        
        register_all_models()
        
        # Try to create EEGNet
        config = {
            'type': 'EEGNet',
            'args': {
                'num_channels': 22,
                'num_classes': 4,
                'num_bands': 1,
                'input_length': 750
            }
        }
        
        model = build_from_config(config, MODELS)
        
        # Test forward pass
        dummy_input = torch.randn(2, 1, 22, 750)
        output = model(dummy_input)
        
        print(f"  [OK] EEGNet created and tested")
        print(f"    Input shape: {dummy_input.shape}")
        print(f"    Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_directory_structure():
    """Test that all required directories exist."""
    print("\nTesting directory structure...")
    
    required_dirs = [
        'core', 'core/base',
        'data', 'data/datasets',
        'models',
        'trainers',
        'utils',
        'configs', 'configs/experiment', 'configs/dataset',
        'experiments',
        'scripts'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"  [OK] {dir_path}/")
        else:
            print(f"  [FAIL] {dir_path}/ (missing)")
            all_exist = False
    
    return all_exist


def main():
    """Run all tests."""
    print("=" * 60)
    print("EEG-BCI Framework Test Suite")
    print("=" * 60)
    
    results = []
    
    results.append(("Directory Structure", test_directory_structure()))
    results.append(("Imports", test_imports()))
    results.append(("Registrations", test_registrations()))
    results.append(("Configuration", test_config()))
    results.append(("Model Creation", test_model_creation()))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: {name}")
    
    all_passed = all(r[1] for r in results)
    
    print("=" * 60)
    if all_passed:
        print("All tests passed!")
        return 0
    else:
        print("Some tests failed. Please check the output above.")
        for name, passed in results:
            if not passed:
                print(f"  Failed: {name}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
