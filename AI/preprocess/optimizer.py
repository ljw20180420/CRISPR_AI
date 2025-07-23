from typing import Literal
from torch import nn
from transformers import PreTrainedModel
from transformers.trainer_pt_utils import get_parameter_names


class MyOptimizer:
    def __init__(
        self,
        name: Literal["adamw_torch", "adamw_torch_fused", "adafactor"],
        learning_rate: float,
        weight_decay: float,
        model: PreTrainedModel,
    ):
        """Parameters of optimizer.

        Args:
            name: Name of optimizer.
            learning_rate: Learn rate of the optimizer.
            weight_decay: The l2 regularization coefficient.
        """
        self.name = name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        forbidden_name_patterns = [
            r"bias",
            r"layernorm",
            r"rmsnorm",
            r"(?:^|\.)norm(?:$|\.)",
            r"_norm(?:$|\.)",
        ]
        decay_parameters = get_parameter_names(
            model, [nn.LayerNorm], forbidden_name_patterns
        )
        params = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]

        if self.name == "adafactor":
            from transformers.optimization import Adafactor

            self.optimizer = Adafactor(
                params=params,
                lr=self.learning_rate,
                scale_parameter=False,
                relative_step=False,
            )

        elif self.name in ["adamw_torch", "adamw_torch_fused"]:
            from torch.optim import AdamW

            fused = True if self.name == "adamw_torch_fused" else False
            self.optimizer = AdamW(
                params=params,
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                fused=fused,
            )
