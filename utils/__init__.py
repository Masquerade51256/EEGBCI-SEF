"""
Utility functions for the EEG-BCI experiment framework.
"""

from .path_manager import PathManager
from .logging import setup_logger
from .visualization import plot_training_history, plot_subject_comparison

__all__ = [
    'PathManager',
    'setup_logger',
    'plot_training_history',
    'plot_subject_comparison',
]
