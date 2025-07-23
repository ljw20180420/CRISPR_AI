from typing import Literal
import numpy as np
from transformers.optimization import get_scheduler
from .optimizer import MyOptimizer


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
        num_training_steps: int,
        my_optimizer: MyOptimizer,
    ) -> None:
        """Parameters for learning rate scheduler.

        Args:
            name: The scheduler type to use.
            warmup_ratio: Ratio of total training steps used for a linear warmup from 0 to learning_rate.
        """
        self.name = name
        self.warmup_ratio = warmup_ratio
        self.num_training_steps = num_training_steps
        self.lr_scheduler = get_scheduler(
            name=self.name,
            optimizer=my_optimizer.optimizer,
            num_warmup_steps=int(np.ceil(num_training_steps * self.warmup_ratio)),
            num_training_steps=num_training_steps,
            scheduler_specific_kwargs={},
        )
