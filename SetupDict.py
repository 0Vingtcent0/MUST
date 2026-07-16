import torch.optim as optim
import core
from train_tools import models

__all__ = ["TRAINER", "OPTIMIZER", "MODELS", "PREDICTOR"]

TRAINER = {
    "must": core.MUST.Trainer,
}

PREDICTOR = {
    "must": core.MUST.Predictor,
}

MODELS = {
    "must": models.MUST,
}

OPTIMIZER = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
    "adamw": optim.AdamW,
}

