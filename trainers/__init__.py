"""Trainer implementations."""

from .supervised_trainer import SupervisedTrainer
from .transfer_learning_trainer import TransferLearningTrainer
from .loso_trainer import LOSOTrainer
from .streaming_loso_trainer import StreamingLOSOTrainer
from .logo_trainer import LOGOTrainer
from .dann_trainer import DANNStreamingLOSOTrainer
from .clinical_loso_trainer import ClinicalStreamingLOSOTrainer
from core.registry import TRAINERS

# Register the trainers
TRAINERS.register('SupervisedTrainer')(SupervisedTrainer)
TRAINERS.register('TransferLearningTrainer')(TransferLearningTrainer)
TRAINERS.register('LOSOTrainer')(LOSOTrainer)
TRAINERS.register('StreamingLOSOTrainer')(StreamingLOSOTrainer)
TRAINERS.register('LOGOTrainer')(LOGOTrainer)
TRAINERS.register('DANNStreamingLOSOTrainer')(DANNStreamingLOSOTrainer)
TRAINERS.register('ClinicalStreamingLOSOTrainer')(ClinicalStreamingLOSOTrainer)

__all__ = ['SupervisedTrainer', 'TransferLearningTrainer', 'LOSOTrainer',
           'StreamingLOSOTrainer', 'LOGOTrainer', 'DANNStreamingLOSOTrainer',
           'ClinicalStreamingLOSOTrainer']
