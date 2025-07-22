from typing import Literal
import numpy as np
from datasets import Dataset
from torch.optim import Optimizer
from transformers.optimization import get_scheduler


class MyLRScheduler:
    def __init__(
        self,
        name: Literal[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
            "inverse_sqrt",
            "reduce_lr_on_plateau",
            "cosine_with_min_lr",
            "warmup_stable_decay",
        ],
        warmup_ratio: float,
    ) -> None:
        """Parameters for learning rate scheduler.

        Args:
            name: The scheduler type to use.
            warmup_ratio: Ratio of total training steps used for a linear warmup from 0 to learning_rate.
        """
        self.name = name
        self.warmup_ratio = warmup_ratio

    def __call__(
        self, dataset: Dataset, batch_size: int, num_epochs: int, optimizer: Optimizer
    ) -> None:
        num_training_steps = np.ceil(len(dataset) / batch_size) * num_epochs
        self.lr_scheduler = get_scheduler(
            name=self.name,
            optimizer=optimizer,
            num_warmup_steps=np.ceil(num_training_steps * self.warmup_ratio),
            num_training_steps=num_training_steps,
            scheduler_specific_kwargs={},
        )
