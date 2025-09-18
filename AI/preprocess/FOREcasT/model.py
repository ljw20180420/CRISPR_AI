import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Optional

# torch does not import opt_einsum as backend by default. import opt_einsum manually will enable it.
from torch.backends import opt_einsum
from einops import rearrange, einsum, repeat
from .data_collator import DataCollator
from common_ai.utils import MyGenerator


class FOREcasTModel(nn.Module):
    model_type = "FOREcasT"

    def __init__(
        self,
        max_del_size: int,
        reg_const: float,
        i1_reg_const: float,
    ) -> None:
        """FOREcasT arguments.

        Args:
            max_del_size: maximal deletion size.
            reg_const: regularization coefficient for insertion.
            i1_reg_const: regularization coefficient for deletion.
        """
        super().__init__()

        self.data_collator = DataCollator(max_del_size=max_del_size)
        is_delete = torch.tensor(
            ["I" not in label for label in self._get_feature_label()]
        )
        self.register_buffer(
            "reg_coff",
            (is_delete * reg_const + ~is_delete * i1_reg_const),
        )
        self.linear = nn.Linear(
            in_features=len(self.reg_coff), out_features=1, bias=False
        )

    def _get_feature_label(self) -> list[str]:
        feature_DelSize_label = ["Any Deletion", "D1", "D2-3", "D4-7", "D8-12", "D>12"]
        feature_InsSize_label = ["Any Insertion", "I1", "I2"]
        feature_DelLoc_label = [
            "DL-1--1",
            "DL-2--2",
            "DL-3--3",
            "DL-4--6",
            "DL-7--10",
            "DL-11--15",
            "DL-16--30",
            "DL<-30",
            "DL>=0",
            "DR0-0",
            "DR1-1",
            "DR2-2",
            "DR3-5",
            "DR6-9",
            "DR10-14",
            "DR15-29",
            "DR<0",
            "DR>=30",
        ]
        feature_InsSeq_label = [
            "I1_A",
            "I1_C",
            "I1_G",
            "I1_T",
            "I2_AA",
            "I2_AC",
            "I2_AG",
            "I2_AT",
            "I2_CA",
            "I2_CC",
            "I2_CG",
            "I2_CT",
            "I2_GA",
            "I2_GC",
            "I2_GG",
            "I2_GT",
            "I2_TA",
            "I2_TC",
            "I2_TG",
            "I2_TT",
        ]
        feature_InsLoc_label = ["IL-1--1", "IL-2--2", "IL-3--3", "IL<-3", "IL>=0"]
        feature_LocalCutSiteSequence_label = []
        for offset in range(-5, 4):
            for nt in ["A", "G", "C", "T"]:
                feature_LocalCutSiteSequence_label.append(f"CS{offset}_NT={nt}")
        feature_LocalCutSiteSeqMatches_label = []
        for offset1 in range(-3, 2):
            for offset2 in range(-3, offset1):
                for nt in ["A", "G", "C", "T"]:
                    feature_LocalCutSiteSeqMatches_label.append(
                        f"M_CS{offset1}_{offset2}_NT={nt}"
                    )
        feature_LocalRelativeSequence_label = []
        for offset in range(-3, 3):
            for nt in ["A", "G", "C", "T"]:
                feature_LocalRelativeSequence_label.append(f"L{offset}_NT={nt}")
        for offset in range(-3, 3):
            for nt in ["A", "G", "C", "T"]:
                feature_LocalRelativeSequence_label.append(f"R{offset}_NT={nt}")
        feature_SeqMatches_label = []
        for loffset in range(-3, 3):
            for roffset in range(-3, 3):
                feature_SeqMatches_label.append(f"X_L{loffset}_R{roffset}")
                feature_SeqMatches_label.append(f"M_L{loffset}_R{roffset}")
        feature_I1or2Rpt_label = ["I1Rpt", "I1NonRpt", "I2Rpt", "I2NonRpt"]
        feature_microhomology_label = [
            "L_MH1-1",
            "R_MH1-1",
            "L_MH2-2",
            "R_MH2-2",
            "L_MH3-3",
            "R_MH3-3",
            "L_MM1_MH3-3",
            "R_MM1_MH3-3",
            "L_MH4-6",
            "R_MH4-6",
            "L_MM1_MH4-6",
            "R_MM1_MH4-6",
            "L_MH7-10",
            "R_MH7-10",
            "L_MM1_MH7-10",
            "R_MM1_MH7-10",
            "L_MH11-15",
            "R_MH11-15",
            "L_MM1_MH11-15",
            "R_MM1_MH11-15",
            "No MH",
        ]
        return (
            self._features_pairwise_label(feature_DelSize_label, feature_DelLoc_label)
            + feature_InsSize_label
            + feature_DelSize_label
            + feature_DelLoc_label
            + feature_InsLoc_label
            + feature_InsSeq_label
            + self._features_pairwise_label(
                feature_LocalCutSiteSequence_label,
                feature_InsSize_label + feature_DelSize_label,
            )
            + self._features_pairwise_label(
                feature_microhomology_label + feature_LocalRelativeSequence_label,
                feature_DelSize_label + feature_DelLoc_label,
            )
            + self._features_pairwise_label(
                feature_LocalCutSiteSeqMatches_label + feature_SeqMatches_label,
                feature_DelSize_label,
            )
            + self._features_pairwise_label(
                feature_InsSeq_label
                + feature_LocalCutSiteSequence_label
                + feature_LocalCutSiteSeqMatches_label,
                feature_I1or2Rpt_label,
            )
            + feature_I1or2Rpt_label
            + feature_LocalCutSiteSequence_label
            + feature_LocalCutSiteSeqMatches_label
            + feature_LocalRelativeSequence_label
            + feature_SeqMatches_label
            + feature_microhomology_label
        )

    def _features_pairwise_label(
        self, features1_label: list[str], features2_label: list[str]
    ) -> list[str]:
        features_label = []
        for label1 in features1_label:
            for label2 in features2_label:
                features_label.append(f"PW_{label1}_vs_{label2}")
        return features_label

    def forward(
        self, input: dict, label: Optional[dict], my_generator: MyGenerator
    ) -> dict:
        logit = rearrange(self.linear(input["feature"].to(self.device)), "b m 1 -> b m")
        if label is not None:
            loss, loss_num = self.loss_fun(
                logit,
                label["count"].to(self.device),
            )
            return {
                "logit": logit,
                "loss": loss,
                "loss_num": loss_num,
            }
        return {"logit": logit}

    def loss_fun(self, logit: torch.Tensor, count: torch.Tensor) -> float:
        # kl divergence
        batch_size = logit.shape[0]
        loss = F.kl_div(
            F.log_softmax(logit, dim=1),
            F.normalize(
                count + 0.5, p=1.0, dim=1
            ),  # add 0.5 to prevent log(0), see loadOligoFeaturesAndReadCounts
            reduction="sum",
        ) + batch_size * einsum(self.reg_coff, self.linear.weight**2, "f, o f ->")
        loss_num = batch_size
        return loss, loss_num

    def eval_output(self, examples: list[dict], batch: dict) -> pd.DataFrame:
        result = self(input=batch["input"], label=None, my_generator=None)

        probas = F.softmax(result["logit"], dim=1).cpu().numpy()
        batch_size, feature_size = probas.shape

        left_shift = np.zeros((batch_size, 20), dtype=int)
        right_shift = np.zeros((batch_size, 20), dtype=int)
        ins_shift = np.full((batch_size, 20), "", dtype="U2")
        ins1_2bp = [
            "A",
            "C",
            "G",
            "T",
            "AA",
            "AC",
            "AG",
            "AT",
            "CA",
            "CC",
            "CG",
            "CT",
            "GA",
            "GC",
            "GG",
            "GT",
            "TA",
            "TC",
            "TG",
            "TT",
        ]
        for i, example in enumerate(examples):
            ref = (
                example["ref1"][: example["cut1"]] + example["ref2"][example["cut2"] :]
            )
            cut = example["cut1"]
            for j, ins in enumerate(ins1_2bp):
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

        ins_lens = np.array([len(ins) for ins in ins1_2bp])
        left_shift = np.minimum(ins_lens - right_shift, left_shift)
        rpos1s = np.concatenate(
            [
                repeat(
                    self.data_collator.lefts[:-20],
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
                    self.data_collator.rights[:-20],
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
                    np.arange(batch_size), "b -> (b f)", f=feature_size
                ),
                "proba": probas.flatten(),
                "rpos1": rpos1s.flatten(),
                "rpos2": rpos2s.flatten(),
                "random_ins": random_ins.flatten(),
            }
        )
        return df
