"""Trainer implementations."""

from .supervised_trainer import SupervisedTrainer
from core.registry import TRAINERS

# Register the trainer
TRAINERS.register('SupervisedTrainer')(SupervisedTrainer)

__all__ = ['SupervisedTrainer']
