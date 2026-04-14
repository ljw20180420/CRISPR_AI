import importlib
import os
import re
import subprocess

import gradio as gr
import jsonargparse
import pandas as pd
import py2bit
import pysam
import torch
from Bio import Seq
from common_ai.gradio_fn import MyGradioFnAbstract
from common_ai.test import MyTest


class MyGradioFn(MyGradioFnAbstract):
    def __init__(
        self,
        app_cfg: jsonargparse.Namespace,
        train_parser: jsonargparse.ArgumentParser,
    ) -> None:
        # set max_del_size to zero to remove the ref length requirement for inDelphi, Lindel and FOREcasT
        for i in range(len(app_cfg.inference)):
            app_cfg.inference[i].init_args.max_del_size = 0

        breakpoint()
        self.inference_instance_dict = {}
        super().__init__(app_cfg, train_parser)

    @torch.no_grad()
    def __call__(
        self, repo_id: str, spacer: str, eval_output_step: int
    ) -> pd.DataFrame:
        cut = 25
        my_inference = self.inference_instance_dict[repo_id]
        spacer = re.sub(r"[\d\s]", "", spacer)
        if len(spacer) < 20:
            gr.Warning(f"it recommends to provide >=20bp protospacer", duration=None)
        my_inference.model.eval_output_step = eval_output_step

        ref = self.retrieve_ref(spacer)
        cas9 = re.search(r"^CRIfuser_.+_SX_(spcas9|spymac|ispymac)$", repo_id).group(1)
        pam = "GG" if cas9 == "spcas9" else "AA"
        if ref[cut + 4 : cut + 6] != pam:
            gr.Warning(
                f"pam should be N{pam} for {cas9}, but the detected pam is {ref[cut+3:cut+6]}",
                duration=None,
            )

        infer_df = pd.DataFrame(
            {
                "ref": [ref],
                "cut": [cut],
                "scaffold": ["spcas9"],  # dummy, not used by CRIfuser
            }
        )
        infer_out = my_inference(infer_df, test_cfg=None, train_parser=None)
        infer_out = (
            infer_out.sort_values(by="proba", ascending=False)
            .reset_index(drop=True)
            .head(10)
        )
        aligns, indel_types, percents = [], [], []
        for rpos1, rpos2, proba in zip(
            infer_out["rpos1"], infer_out["rpos2"], infer_out["proba"]
        ):
            align_up = ref[: cut + rpos1]
            align_down = ref[cut + rpos2 :]
            mid = "-" * max(0, rpos2 - rpos1)
            aligns.append(align_up + mid + align_down)

            if rpos1 == 0 and rpos2 == 0:
                indel_type = "wild type"
            elif rpos1 <= 0 and rpos2 >= 0:
                indel_type = "deletion"
            elif rpos1 >= 0 and rpos2 <= 0:
                indel_type = "templated insertion"
            else:
                indel_type = "indel"
            indel_types.append(indel_type)

            percent = f"{(proba * 100):.2f}%"
            percents.append(percent)

        result_df = pd.DataFrame(
            {
                ref: aligns,
                "type": indel_types,
                "percent": percents,
            }
        )

        return result_df

    def launch(self):
        cas9_dropdown = []
        for repo_id in self.inference_dict.keys():
            cas9 = re.search(
                r"^CRIfuser_.+_SX_(spcas9|spymac|ispymac)$", repo_id
            ).group(1)
            cas9_dropdown.append((cas9, repo_id))
            self.load_inference(repo_id)

        # warm up
        self(
            repo_id=cas9_dropdown[0][1],
            spacer="CTGGCTTACCTGAATCGTCC",
            eval_output_step=4,
        )

        gr.Interface(
            fn=self,
            inputs=[
                gr.Dropdown(choices=cas9_dropdown, label="select cas9 editors"),
                gr.Textbox(
                    placeholder="CTGGCTTACCTGAATCGTCC",
                    label=">=20bp targeting sequence (protospacer)",
                ),
                gr.Number(
                    value=4,
                    minimum=1,
                    label="sampling step",
                    info="Increase this value to sample less outcomes to calculate the editing profile. This increases speed but decreases accuracy.",
                ),
            ],
            outputs=[
                gr.Dataframe(
                    headers=["outcome", "type", "percent"],
                    datatype=["str", "str", "str"],
                    label="result",
                )
            ],
            description="# Welcome. This app predicts the editing outcomes of G-rich (spycas9) and A-rich (spymac, ispymac) cas9.",
            flagging_mode="never",
        ).launch()

    def load_inference(self, repo_id: str) -> None:
        assert repo_id in self.inference_dict, f"repo id {repo_id} is not found"
        inference_cfg, test_cfg = self.inference_dict[repo_id]
        inference_module, inference_cls = inference_cfg.class_path.rsplit(".", 1)
        self.inference_instance_dict[repo_id] = getattr(
            importlib.import_module(inference_module), inference_cls
        )(**inference_cfg.init_args.as_dict())
        (
            _,
            train_cfg,
            self.inference_instance_dict[repo_id].logger,
            self.inference_instance_dict[repo_id].model,
            self.inference_instance_dict[repo_id].my_generator,
        ) = MyTest(**test_cfg.as_dict()).load_model(self.train_parser)
        self.inference_instance_dict[repo_id].batch_size = train_cfg.train.batch_size

    def retrieve_ref(self, protospacer: str) -> str:
        ext_up = 25
        ext_down = 26
        samfile = self.DEFAULT_TEMP_DIR / os.urandom(16).hex()

        subprocess.run(
            args=[
                "bowtie2",
                "--quiet",
                "-c",
                "-x",
                os.environ["BOWTIE2_INDEX"],
                "-U",
                protospacer,
                "-S",
                samfile,
            ],
        )

        with pysam.AlignmentFile(samfile, "rb") as sam, py2bit.open(
            os.environ["GENOME"]
        ) as tb:
            for align in sam.fetch():
                break
            if not align.is_mapped:
                raise gr.Error("protospacer cannot be mapped", duration=None)
            if align.is_forward:
                start = align.reference_end - 3 - ext_up
                end = align.reference_end - 3 + ext_down
            else:
                start = align.reference_start + 3 - ext_down
                end = align.reference_start + 3 + ext_up

            ref = tb.sequence(align.reference_name, start, end)
            if not align.is_forward:
                ref = str(Seq.Seq(ref).reverse_complement())

        samfile.unlink(missing_ok=True)

        return ref
