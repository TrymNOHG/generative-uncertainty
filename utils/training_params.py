from dataclasses import dataclass

@dataclass
class TrainingParameters:
    learning_rate: float = 1e-3
    batch_size:int = 128
    num_epochs:int = 30