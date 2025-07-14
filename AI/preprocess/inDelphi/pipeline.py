from diffusers import DiffusionPipeline
import torch
from transformers import PreTrainedModel
import torch.nn.functional as F
import numpy as np
import pandas as pd
from .load_data import DataCollator


class inDelphiPipeline(DiffusionPipeline):
    def __init__(
        self,
        core_model: PreTrainedModel,
        auxilary_model: PreTrainedModel,
    ) -> None:
        super().__init__()

        self.register_modules(core_model=core_model)
        self.register_modules(auxilary_model=auxilary_model)
        self.data_collator = DataCollator(
            DELLEN_LIMIT=core_model.config.DELLEN_LIMIT,
            output_label=True,
        )

    @torch.no_grad()
    def __call__(self, examples: list[dict], output_label: bool) -> pd.DataFrame:
        self.data_collator.output_label = output_label
        batch = self.data_collator(examples)
        if output_label:
            result = self.core_model(
                batch["mh_input"].to(self.core_model.device),
                batch["mh_del_len"].to(self.core_model.device),
                batch["genotype_count"].to(self.core_model.device),
                batch["total_del_len_count"].to(self.core_model.device),
            )
        else:
            result = self.core_model(
                batch["mh_input"].to(self.core_model.device),
                batch["mh_del_len"].to(self.core_model.device),
            )
        insert_batch = self.data_collator.insert_call(examples)
        insert_probabilities, insert_1bps = self.auxilary_model(
            result["total_del_len_weight"],
            insert_batch["onebp_feature"],
            insert_batch["m654"],
            use_m654=False,
        )
        batch_size = result["mh_weight"].shape[0]
        rightests, mh_mh_lens, mh_del_lens = self.data_collator.get_auxilaries(examples)
        delete_probabilities = F.normalize(
            torch.cat(
                (
                    result["mh_weight"],
                    result["mhless_weight"].expand(batch_size, -1),
                ),
                dim=1,
            ),
            p=1.0,
            dim=1,
        ).cpu().numpy() * (1 - insert_probabilities[:, None])
        DELLEN_LIMIT = self.core_model.config.DELLEN_LIMIT
        sample_idxs, probas, rpos1s, rpos2s, random_inss = [], [], [], [], []
        for sample_idx, (
            delete_probability,
            insert_probability,
            insert_1bp,
            rightest,
            mh_mh_len,
            mh_del_len,
            example,
        ) in enumerate(
            zip(
                delete_probabilities,
                insert_probabilities,
                insert_1bps,
                rightests,
                mh_mh_lens,
                mh_del_lens,
                examples,
            )
        ):
            probas.append(
                (delete_probability[: len(mh_mh_len)] / (mh_mh_len + 1)).repeat(
                    mh_mh_len + 1
                )
            )
            rpos2_mh = np.concatenate(
                [np.arange(rt, rt - ml - 1, -1) for rt, ml in zip(rightest, mh_mh_len)]
            )
            rpos1_mh = rpos2_mh - mh_del_len.repeat(mh_mh_len + 1)
            rpos1s.append(rpos1_mh)
            rpos2s.append(rpos2_mh)
            probas.append(
                (
                    delete_probability[-(DELLEN_LIMIT - 1) :]
                    / np.arange(2, DELLEN_LIMIT + 1)
                ).repeat(np.arange(2, DELLEN_LIMIT + 1))
            )
            rpos1s.append(
                np.concatenate(
                    [np.arange(-DEL_SIZE, 1) for DEL_SIZE in range(1, DELLEN_LIMIT)]
                )
            )
            rpos2s.append(
                np.concatenate(
                    [np.arange(0, DEL_SIZE + 1) for DEL_SIZE in range(1, DELLEN_LIMIT)]
                )
            )
            random_inss.append(
                np.array([""] * (len(probas[-1]) + len(probas[-2])), dtype="1U")
            )
            probas.append(insert_1bp / insert_1bp.sum() * insert_probability)
            tr2 = example["ref2"][example["cut2"] - 1]
            rpos2_ins = np.array(
                [-1 if base == tr2 else 0 for base in ["A", "C", "G", "T"]]
            )
            tr1 = example["ref1"][example["cut1"]]
            rpos1_ins = np.array(
                [1 if base == tr1 else 0 for base in ["A", "C", "G", "T"]]
            )
            rpos1_ins = np.minimum(rpos1_ins, 1 + rpos2_ins)
            rpos1s.append(rpos1_ins)
            rpos2s.append(rpos2_ins)
            random_inss.append(
                np.array(
                    [
                        "" if rpos1_ins[i] != 0 or rpos2_ins[i] != 0 else base
                        for i, base in enumerate(["A", "C", "G", "T"])
                    ],
                    dtype="1U",
                )
            )
            sample_idxs.extend(
                (len(probas[-1]) + len(probas[-2]) + len(probas[-3])) * [sample_idx]
            )

        df = (
            pd.DataFrame(
                {
                    "sample_idx": sample_idxs,
                    "proba": np.concatenate(probas, axis=0),
                    "rpos1": np.concatenate(rpos1s, axis=0),
                    "rpos2": np.concatenate(rpos2s, axis=0),
                    "random_ins": np.concatenate(random_inss, axis=0),
                }
            )
            .groupby(by=["sample_idx", "rpos1", "rpos2", "random_ins"])["proba"]
            .sum()
            .reset_index()
        )

        if output_label:
            return (
                df,
                result["loss"],
                batch_size,
            )
        return df

    @torch.no_grad()
    def inference(self, examples: list) -> dict:
        self.data_collator.output_label = False
        return self.__call__(
            examples=self.data_collator.inference(examples),
            output_label=False,
        )
