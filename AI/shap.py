from typing import Literal
import numpy as np
import pandas as pd
import datasets
import jsonargparse

from common_ai.shap import MyShapAbstract
from common_ai.utils import SeqTokenizer
from AI.inference import MyInference
from AI.preprocess.utils import MicroHomologyTool


class MyShap(MyShapAbstract):

    def __init__(
        self,
        explainer_cls: Literal["SamplingExplainer"],
        load_only: bool,
        shap_target: Literal["small_indel", "unilateral", "large_indel", "mmej"],
        nsamples_per_feature: int,
        seed: int,
    ):
        """SHAP arguments.

        Args:
            explainer_cls: the model agnostic explainer method.
            load_only: only load existing explanation.
            shap_target: shap target.
            nsamples_per_feature: number of sampling for each feature while explaining.
            seed: seed for reproducibility.
        """

        super().__init__(
            explainer_cls, load_only, shap_target, nsamples_per_feature, seed
        )
        self.DNA_tokenizer = SeqTokenizer("ACGT")
        self.micro_homology_tool = MicroHomologyTool()

    def dataset2pandas(
        self, ds_list: list[datasets.Dataset], my_inference: MyInference
    ) -> pd.DataFrame:
        samples = []
        for ds in ds_list:
            for i in range(ds.num_rows):
                example = ds[i]
                ref1 = example["ref1"]
                ref2 = example["ref2"]
                cut1 = example["cut1"]
                cut2 = example["cut2"]
                scaffold = example["scaffold"]
                if cut1 >= my_inference.ext_up:
                    ref_up = ref1[cut1 - my_inference.ext_up : cut1]
                else:
                    assert (
                        cut2 >= my_inference.ext_up
                    ), f"sequence upstream to cut should be at least {self.ext_up} bps"
                    ref_up = ref2[cut2 - my_inference.ext_up : cut2]
                if len(ref2) - cut2 >= my_inference.ext_down:
                    ref_down = ref2[cut2 : cut2 + my_inference.ext_down]
                else:
                    assert (
                        len(ref1) - cut1 >= my_inference.ext_down
                    ), f"sequence downstream to cut should be at least {self.ext_down} bps"
                    ref_down = ref1[cut1 : cut1 + my_inference.ext_down]
                ref = ref_up + ref_down

                samples.append(
                    np.append(
                        self.DNA_tokenizer(ref),
                        my_inference.scaffolds["spymac"] == scaffold,
                    )
                )

        return pd.DataFrame(
            data=np.array(samples),
            columns=[
                f"pos{i}" for i in range(my_inference.ext_up + my_inference.ext_down)
            ]
            + ["spymac"],
        )

    def predict(
        self,
        X: pd.DataFrame,
        my_inference: MyInference,
        test_cfg: jsonargparse.Namespace,
        train_parser: jsonargparse.ArgumentParser,
    ) -> pd.DataFrame:
        infer_in = self.numpy2infer_in(X, my_inference)
        infer_out = my_inference(
            infer_df=infer_in,
            test_cfg=test_cfg,
            train_parser=train_parser,
        )
        return self.infer_out2shap(infer_out, infer_in)

    # explainer will convert pandas to numpy
    def numpy2infer_in(
        self, arr: np.ndarray, my_inference: MyInference
    ) -> pd.DataFrame:
        char_arr = np.array(["A", "C", "G", "T"])[arr[:, :-1].astype(dtype=np.int8)]
        return pd.DataFrame(
            {
                "ref": ["".join(char_row) for char_row in char_arr],
                "cut": [my_inference.ext_up] * arr.shape[0],
                "scaffold": [
                    "spymac" if spymac > 0 else "spcas9" for spymac in arr[:, -1]
                ],
            }
        )

    def infer_out2shap(
        self,
        infer_out: pd.DataFrame,
        infer_in: pd.DataFrame,
    ) -> np.ndarray:
        # prevent query out all rows for a sample index
        sample_num = infer_out["sample_idx"].max() + 1
        zero_probas = pd.Series(np.zeros(sample_num), name="proba")

        if self.shap_target == "small_indel":
            probas = (
                infer_out.assign(
                    abs_rpos1=lambda df: df["rpos1"].abs(),
                    abs_rpos2=lambda df: df["rpos2"].abs(),
                )
                .query("abs_rpos1 <= 2 and abs_rpos2 <= 2")
                .groupby("sample_idx")
                .agg({"proba": "sum"})["proba"]
            )

        elif self.shap_target == "unilateral":
            probas = (
                infer_out.assign(
                    abs_rpos1=lambda df: df["rpos1"].abs(),
                    abs_rpos2=lambda df: df["rpos2"].abs(),
                )
                .query(
                    "abs_rpos1 <= 2 and abs_rpos2 > 2 or abs_rpos1 > 2 and abs_rpos2 <= 2"
                )
                .groupby("sample_idx")
                .agg({"proba": "sum"})["proba"]
            )

        elif self.shap_target == "large_indel":
            probas = (
                infer_out.assign(
                    abs_rpos1=lambda df: df["rpos1"].abs(),
                    abs_rpos2=lambda df: df["rpos2"].abs(),
                )
                .query("abs_rpos1 > 2 and abs_rpos2 > 2")
                .groupby("sample_idx")
                .agg({"proba": "sum"})["proba"]
            )

        elif self.shap_target == "mmej":
            mh_matrices = []
            for ref, cut in zip(infer_in["ref"], infer_in["cut"]):
                self.micro_homology_tool.reinitialize(ref1=ref, ref2=ref)
                mh_matrix, _, _, _ = self.micro_homology_tool.get_mh(
                    ref1=ref, ref2=ref, cut1=cut, cut2=cut, ext1=0, ext2=0
                )
                mh_matrices.append(mh_matrix.reshape(len(ref) + 1, len(ref) + 1))

            infer_out["mh"] = np.array(mh_matrices)[
                infer_out["sample_idx"],
                infer_out["rpos2"] + cut,
                infer_out["rpos1"] + cut,
            ]

            probas = (
                infer_out.assign(
                    abs_rpos1=lambda df: df["rpos1"].abs(),
                    abs_rpos2=lambda df: df["rpos2"].abs(),
                )
                .query("abs_rpos1 > 2 and abs_rpos2 > 2 and mh > 0")
                .groupby("sample_idx")
                .agg({"proba": "sum"})["proba"]
            )

        return (zero_probas + probas).to_numpy()
