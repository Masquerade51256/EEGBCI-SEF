"""Trainer implementations."""

from .supervised_trainer import SupervisedTrainer
from .transfer_learning_trainer import TransferLearningTrainer
from core.registry import TRAINERS

# Register the trainers
TRAINERS.register('SupervisedTrainer')(SupervisedTrainer)
TRAINERS.register('TransferLearningTrainer')(TransferLearningTrainer)

__all__ = ['SupervisedTrainer', 'TransferLearningTrainer']
