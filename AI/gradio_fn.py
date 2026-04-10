import hashlib
import os
import re
import subprocess
import tempfile

import gradio as gr
import jsonargparse
import pandas as pd
import py2bit
import pysam
import torch
from Bio import Seq
from common_ai.gradio_fn import MyGradioFnAbstract


class MyGradioFn(MyGradioFnAbstract):
    def __init__(
        self,
        app_cfg: jsonargparse.Namespace,
        train_parser: jsonargparse.ArgumentParser,
    ) -> None:
        # set max_del_size to zero to remove the ref length requirement for inDelphi, Lindel and FOREcasT
        for i in range(len(app_cfg.inference)):
            app_cfg.inference[i].init_args.max_del_size = 0

        super().__init__(app_cfg, train_parser)

    @torch.no_grad()
    def __call__(self, repo_id: str, spacer: str) -> tuple[pd.DataFrame, str]:
        cut = 25

        if len(spacer) < 20:
            gr.Warning(f"it recommends to provide >=20bp protospacer")
        self.reload_inference(repo_id=repo_id)
        ref = self.retrieve_ref(spacer)
        cas9 = re.search(r"^CRIfuser_.+_SX_(spcas9|spymac|ispymac)$", repo_id).group(1)
        pam = "GG" if cas9 == "spcas9" else "AA"
        if ref[cut + 4 : cut + 6] == pam:
            gr.Warning(
                f"pam should be N{pam} for {cas9}, but the detected pam is {ref[cut+3:cut+6]}"
            )

        infer_df = pd.DataFrame(
            {
                "ref": [ref],
                "cut": [cut],
                "scaffold": ["spcas9"],  # dummy, not used by CRIfuser
            }
        )
        infer_out = self.my_inference(infer_df, self.test_cfg, self.train_parser)
        infer_out = (
            infer_out.sort_values(by="proba", ascending=False)
            .reset_index(drop=True)
            .head(10)
        )
        aligns = []
        for rpos1, rpos2 in zip(infer_out["rpos1"], infer_out["rpos2"]):
            align_up = ref[: cut + rpos1]
            align_down = ref[cut + rpos2 :]
            mid = "-" * max(0, rpos2 - rpos1)
            aligns.append(align_up + mid + align_down)

        result_df = pd.DataFrame({ref: aligns, "proba": infer_out["proba"]})

        result_df_hash = hashlib.md5(result_df.to_string().encode()).hexdigest()
        temp_dir = self.DEFAULT_TEMP_DIR / result_df_hash
        temp_dir.mkdir(exist_ok=True, parents=True)
        result_df_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir).name
        result_df.to_csv(result_df_file, index=False)

        return result_df, result_df_file

    def launch(self):
        cas9_dropdown = []
        for repo_id in self.inference_dict.keys():
            cas9 = re.search(
                r"^CRIfuser_.+_SX_(spcas9|spymac|ispymac)$", repo_id
            ).group(1)
            cas9_dropdown.append((cas9, repo_id))

        gr.Interface(
            fn=self,
            inputs=[
                gr.Dropdown(choices=cas9_dropdown, label="select cas9 editors"),
                gr.Textbox(
                    placeholder="CTGGCTTACCTGAATCGTCC",
                    label=">=20bp targeting sequence (protospacer)",
                ),
            ],
            outputs=[gr.Dataframe(label="result"), gr.File(label="download result")],
        ).launch()

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
                raise gr.Error("protospacer cannot be mapped")
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
