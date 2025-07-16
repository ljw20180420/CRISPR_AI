from transformers import PretrainedConfig, PreTrainedModel
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Optional, Literal


class LindelConfig(PretrainedConfig):
    model_type = "Lindel"
    label_names = ["count"]

    def __init__(
        self,
        dlen: Optional[int] = None,
        mh_len: Optional[int] = None,
        reg_mode: Optional[Literal["l2", "l1"]] = None,
        reg_const: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """Lindel parameters

        Args:
            dlen: the upper limit of deletion length (strictly less than dlen).
            mh_len: the upper limit of micro-homology length.
            reg_model: regularization method, should be l2 or l1.
            reg_const: regularization coefficient.
        """
        self.dlen = dlen
        self.mh_len = mh_len
        self.reg_mode = reg_mode
        self.reg_const = reg_const
        self.seed = seed
        super().__init__(**kwargs)


class LindelModel(PreTrainedModel):
    config_class = LindelConfig

    def __init__(self, config: LindelConfig) -> None:
        super().__init__(config)
        # In more recent versions of PyTorch, you no longer need to explicitly register_parameter, it's enough to set a member of your nn.Module with nn.Parameter to "notify" pytorch that this variable should be treated as a trainable parameter (https://stackoverflow.com/questions/59234238/how-to-add-parameters-in-module-class-in-pytorch-custom-model).
        self.generator = torch.Generator().manual_seed(config.seed)
        self.reg_mode = config.reg_mode
        self.reg_const = config.reg_const
        # onehotencoder(ref[cut-17:cut+3])
        self.model_indel = nn.Linear(in_features=20 * 4 + 19 * 16, out_features=2)
        # onehotencoder(ref[cut-3:cut+3])
        self.model_ins = nn.Linear(in_features=6 * 4 + 5 * 16, out_features=21)
        # concatenate get_feature and onehotencoder(ref[cut-17:cut+3])
        class_dim = (5 + 1 + 5 + config.dlen - 1) * (config.dlen - 1) // 2
        self.model_del = nn.Linear(
            in_features=class_dim * (config.mh_len + 1) + 20 * 4 + 19 * 16,
            out_features=class_dim,
        )

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1, generator=self.generator)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=1, generator=self.generator)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0, std=1, generator=self.generator)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input: dict, count: Optional[dict] = None) -> torch.Tensor:
        logit_indel = self.model_indel(input["input_indel"])
        logit_ins = self.model_ins(input["input_ins"])
        logit_del = self.model_del(input["input_del"])
        if count is not None:
            loss = (
                self.cross_entropy_reg(
                    logit_indel, count["count_indel"], self.model_indel
                )
                + self.cross_entropy_reg(logit_ins, count["count_ins"], self.model_ins)
                + self.cross_entropy_reg(logit_del, count["count_del"], self.model_del)
            )
            return {
                "logit_indel": logit_indel,
                "logit_ins": logit_ins,
                "logit_del": logit_del,
                "loss": loss,
            }
        return {
            "logit_indel": logit_indel,
            "logit_ins": logit_ins,
            "logit_del": logit_del,
        }

    def cross_entropy_reg(
        self, logit: torch.Tensor, count: torch.Tensor, linear: nn.Linear
    ) -> float:
        if self.reg_mode == "l2":
            reg_term = (linear.weight**2).sum()
        elif self.reg_mode == "l1":
            reg_term = abs(linear.weight).sum()
        return (
            -(
                F.log_softmax(logit, dim=1)
                * F.normalize(count.to(torch.float32), p=1.0, dim=1)
            ).sum()
            + logit.shape[0] * self.reg_const * reg_term
        )
