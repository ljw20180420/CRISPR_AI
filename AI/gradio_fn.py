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


class MyGradioFn(MyGradioFnAbstract):
    def __init__(
        self,
        app_cfg: jsonargparse.Namespace,
        train_parser: jsonargparse.ArgumentParser,
    ) -> None:
        # set max_del_size to zero to remove the ref length requirement for inDelphi, Lindel and FOREcasT
        for i in range(len(app_cfg.inference)):
            app_cfg.inference[i].init_args.max_del_size = 0

        self.inference_instance_dict = {}
        super().__init__(app_cfg, train_parser)

    @torch.no_grad()
    def __call__(self, repo_id: str, spacer: str) -> pd.DataFrame:
        cut = 25
        my_inference = self.inference_instance_dict[repo_id]
        spacer = re.sub(r"[\d\s]", "", spacer)
        if len(spacer) < 20:
            gr.Warning("it recommends to provide >=20bp protospacer", duration=None)

        ref = self.retrieve_ref(spacer)
        cas9 = re.search(r"^CRIfuser_.+_SX_(spcas9|spymac|ispymac)$", repo_id).group(1)
        pam = "GG" if cas9 == "spcas9" else "AA"
        if ref[cut + 4 : cut + 6] != pam:
            gr.Warning(
                f"pam should be N{pam} for {cas9}, but the detected pam is {ref[cut + 3 : cut + 6]}",
                duration=None,
            )

        infer_df = pd.DataFrame({
            "ref": [ref],
            "cut": [cut],
            "scaffold": ["spcas9"],  # dummy, not used by CRIfuser
        })
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
            align = align_up + mid + align_down
            align = align[:cut] + "|" + align[cut:]
            aligns.append(align)

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

        result_df = pd.DataFrame({
            ref[:cut] + "|" + ref[cut:]: aligns,
            "type": indel_types,
            "percent": percents,
        })

        return result_df

    def launch(self):
        for repo_id in self.inference_dict.keys():
            self.inference_instance_dict[repo_id] = self.load_inference(repo_id)

        gr.Interface(
            fn=self,
            inputs=[
                gr.Radio(
                    choices=[
                        ("spycas9", "CRIfuser_CRIfuser_SX_spcas9"),
                        ("spymac", "CRIfuser_CRIfuser_SX_spymac"),
                        ("ispymac", "CRIfuser_CRIfuser_SX_ispymac"),
                    ],
                    label="select cas9 editors",
                    info="Spycas9 for NGG PAM. (i)spymac for NAA PAM.",
                ),
                gr.Textbox(
                    placeholder="CTGGCTTACCTGAATCGTCC",
                    label=">=20bp targeting sequence (protospacer)",
                    info="The protospacer is search on human hg19 genome to determine the complete targeting site. No matter how long the provided protospacer is, PAM always follows the 3' base of the protospacer.",
                ),
            ],
            outputs=[
                gr.Dataframe(
                    headers=["outcome", "type", "percent"],
                    datatype=["str", "str", "str"],
                    label="result",
                    buttons=["copy", "fullscreen"],
                    show_search="filter",
                ),
            ],
            examples=[
                ["CRIfuser_CRIfuser_SX_spcas9", "GAAACAAACAAGAAGAAGCG"],
                ["CRIfuser_CRIfuser_SX_spymac", "ATTCCTGAATCTAGACTCCT"],
                ["CRIfuser_CRIfuser_SX_ispymac", "ATTGAATGTAGATATCCTAT"],
            ],
            cache_examples=True,
            cache_mode="eager",
            description="""
# Welcome. This app predicts the editing outcomes of G-rich (spycas9) and A-rich (spymac, ispymac) cas9.

We search the protospacer in human hg19 genome. If the protospacer is not mapped successfully, an error occurs. If the protospacer is mapped, but PAM next to the protospacer in the human genome does not match the selected cas9 editor (NGG for spycas9, NAA for spymac/ispymac), then only a warning occurs, and the app will predict a result anyway. You can try examples.
            """,
            flagging_mode="never",
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

        with (
            pysam.AlignmentFile(samfile, "rb") as sam,
            py2bit.open(os.environ["GENOME"]) as tb,
        ):
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
