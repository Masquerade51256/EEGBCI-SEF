#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from core import Config, ExperimentManager
from models import register_all_models
from data import DATASETS
from trainers import TRAINERS

register_all_models()

# Load config
config = Config.fromfile('configs/experiment/xw_adfcnn.yaml')
config.set('data.subjects', [1])
config.set('training.epochs', 1)
config.set('training.batch_size', 16)
config.set('training.device', 'cpu')
config.set('experiment.name', 'xw_adfcnn_cpu_test')

print('Starting experiment with CPU...')
exp = ExperimentManager(config, exp_name='xw_adfcnn_cpu_test')
exp.setup()
exp.build_dataset()
print(f"Dataset loaded: {exp.datasets['instances'][1].data.shape}")
exp.build_model()
print(f"Model built: {type(exp.model).__name__}")
exp.build_trainer()
print('Running training (1 epoch, 1 subject)...')
results = exp.run()
print(f"Success! Results: {results}")
