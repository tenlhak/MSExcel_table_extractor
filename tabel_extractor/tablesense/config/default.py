from dataclasses import dataclass

@dataclass
class ModelConfig:
    input_channels: int = 20
    hidden_dim: int = 64
    roi_size: int = 7
    nms_threshold: float = 0.7
    score_threshold: float = 0.5

@dataclass
class TrainingConfig:
    batch_size: int = 4
    learning_rate: float = 0.001
    num_epochs: int = 50
    num_workers: int = 8