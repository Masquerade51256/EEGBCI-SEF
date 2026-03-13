import os
DATASETS = []


class Config:
    def __init__(self):
        self.data_path = os.getenv('DATA_PATH', './data')
        self.model_path = os.getenv('MODEL_PATH', './model')
        self.batch_size = int(os.getenv('BATCH_SIZE', 32))
        self.learning_rate = float(os.getenv('LEARNING_RATE', 0.001))
        self.num_epochs = int(os.getenv('NUM_EPOCHS', 10))