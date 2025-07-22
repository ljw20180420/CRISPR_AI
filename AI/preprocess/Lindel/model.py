import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Optional, Literal

# torch does not import opt_einsum as backend by default. import opt_einsum manually will enable it.
from torch.backends import opt_einsum
from einops import einsum, repeat
from ..model import BaseModel, BaseConfig


class LindelConfig(BaseConfig):
    model_type = "Lindel"

    def __init__(
        self,
        dlen: int,
        mh_len: int,
        reg_mode: Literal["l2", "l1"],
        reg_const: float,
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
        super().__init__(**kwargs)


class LindelModel(BaseModel):
    config_class = LindelConfig

    def __init__(self, config: LindelConfig) -> None:
        super().__init__(config)
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

        self._initialize_model_layer_weights()

    def forward(self, input: dict, label: Optional[dict] = None) -> dict:
        logit_indel = self.model_indel(input["input_indel"])
        logit_ins = self.model_ins(input["input_ins"])
        logit_del = self.model_del(input["input_del"])
        if label is not None:
            loss = self.loss_fun(
                logit_indel=logit_indel,
                count_indel=label["count_indel"],
                model_indel=self.model_indel,
                logit_ins=logit_ins,
                count_ins=label["count_ins"],
                model_ins=self.model_ins,
                logit_del=logit_del,
                count_del=label["count_del"],
                model_del=self.model_del,
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

    def loss_fun(
        self,
        logit_indel: torch.Tensor,
        count_indel: torch.Tensor,
        model_indel: nn.Linear,
        logit_ins: torch.Tensor,
        count_ins: torch.Tensor,
        model_ins: nn.Linear,
        logit_del: torch.Tensor,
        count_del: torch.Tensor,
        model_del: nn.Linear,
    ) -> float:
        return (
            self._cross_entropy_reg(logit_indel, count_indel, model_indel)
            + self._cross_entropy_reg(logit_ins, count_ins, model_ins)
            + self._cross_entropy_reg(logit_del, count_del, model_del)
        )

    def _cross_entropy_reg(
        self, logit: torch.Tensor, count: torch.Tensor, linear: nn.Linear
    ) -> float:
        if self.reg_mode == "l2":
            reg_term = (linear.weight**2).sum()
        elif self.reg_mode == "l1":
            reg_term = abs(linear.weight).sum()
        batch_size = logit.shape[0]
        return (
            -einsum(
                F.log_softmax(logit, dim=1),
                F.normalize(count.to(torch.float32), p=1.0, dim=1),
                "b m, b m ->",
            )
            + batch_size * self.reg_const * reg_term
        )

    def eval_output(self, examples: list[dict], batch: dict) -> pd.DataFrame:
        result = self(batch["input"])

        batch_size = len(examples)
        left_shift = np.zeros((batch_size, 20), dtype=int)
        right_shift = np.zeros((batch_size, 20), dtype=int)
        ins_shift = np.full((batch_size, 20), "", dtype="U2")
        for i, example in enumerate(examples):
            ref = (
                example["ref1"][: example["cut1"]] + example["ref2"][example["cut2"] :]
            )
            cut = example["cut1"]
            for j, ins in enumerate(self.inss):
                if ref[cut] == ins[0]:
                    if len(ins) == 2 and ref[cut + 1] == ins[1]:
                        left_shift[i, j] = 2
                    else:
                        left_shift[i, j] = 1
                if ref[cut - 1] == ins[-1]:
                    if len(ins) == 2 and ref[cut - 2] == ins[-2]:
                        right_shift[i, j] = 2
                    else:
                        right_shift[i, j] = 1
                ins_shift[i, j] = ins[left_shift[i, j] : (len(ins) - right_shift[i, j])]

        ins_lens = np.array([len(ins) for ins in self.inss])
        left_shift = np.minimum(ins_lens - right_shift, left_shift)

        indel_probas = F.softmax(result["logit_indel"], dim=1).cpu().numpy()
        ins_probas = F.softmax(result["logit_ins"], dim=1).cpu().numpy()
        del_probas = F.softmax(result["logit_del"], dim=1).cpu().numpy()
        long_ins_proba = ins_probas[:, -1] * indel_probas[:, 1]
        indel_probas[:, 1] = indel_probas[:, 1] - long_ins_proba
        indel_probas[:, 0] = indel_probas[:, 0] / (1 - long_ins_proba)
        indel_probas[:, 1] = indel_probas[:, 1] / (1 - long_ins_proba)
        ins_probas = ins_probas[:, :-1]

        probas = np.concatenate(
            [
                del_probas * indel_probas[:, [0]],
                ins_probas * indel_probas[:, [1]],
            ],
            axis=1,
        )
        batch_size, feature_size = probas.shape
        rpos1s = np.concatenate(
            [
                repeat(
                    np.concatenate(
                        [np.arange(-dl - 2, 3) for dl in range(1, self.dlen)]
                    ),
                    "f -> b f",
                    b=batch_size,
                ),
                left_shift,
            ],
            axis=1,
        )
        rpos2s = np.concatenate(
            [
                repeat(
                    np.concatenate(
                        [np.arange(-2, dl + 3) for dl in range(1, self.dlen)]
                    ),
                    "f -> b f",
                    b=batch_size,
                ),
                -right_shift,
            ],
            axis=1,
        )
        random_ins = np.concatenate(
            [
                np.full((batch_size, feature_size - 20), "", dtype="U2"),
                ins_shift,
            ],
            axis=1,
        )
        df = pd.DataFrame(
            {
                "sample_idx": repeat(
                    np.arange(batch_size),
                    "b -> (b f)",
                    f=feature_size,
                ),
                "proba": probas.flatten(),
                "rpos1": rpos1s.flatten(),
                "rpos2": rpos2s.flatten(),
                "random_ins": random_ins.flatten(),
            }
        )

        return df
