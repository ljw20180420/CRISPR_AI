import gradio as gr
from diffusers import DiffusionPipeline
import torch
import pandas as pd
from .model import FOREcasTConfig
from .inference import data_collator_inference


@torch.no_grad()
def app(
    data_name: str,
    ref1len: int,
    ref2len: int,
    owner: str,
    device: str,
) -> None:
    pipe = DiffusionPipeline.from_pretrained(
        "%s/%s_%s" % (owner, data_name, FOREcasTConfig.model_type),
        trust_remote_code=True,
        custom_pipeline="%s/%s_%s" % (owner, data_name, FOREcasTConfig.model_type),
    )
    pipe.FOREcasT_model.to(device)

    def gradio_fn(ref, cut):
        batch = data_collator_inference(
            examples=[{"ref": ref, "cut": cut}],
            pre_calculated_features=pipe.FOREcasT_model.pre_calculated_features,
            ref1len=ref1len,
            ref2len=ref2len,
            max_del_size=pipe.FOREcasT_model.max_del_size,
        )
        proba, left, right, ins_seq = pipe(batch).values()
        ins_num = 20
        del_num = len(ins_seq) - ins_num

        return pd.DataFrame(
            {
                "Category": ["del"] * del_num + ["ins"] * ins_num,
                "Genotype position": left.tolist(),
                "Inserted Bases": ins_seq,
                "Length": (right - left)[:del_num].tolist()
                + [len(ins) for ins in ins_seq[-ins_num:]],
                "Predicted frequency": proba.tolist(),
                "sequence": [
                    ref[: le + cut] + ins + ref[ri + cut :]
                    for le, ri, ins in zip(left, right, ins_seq)
                ],
            }
        ).sort_values(by="Predicted frequency", ascending=False)

    gr.Interface(
        fn=gradio_fn,
        inputs=["text", "number"],
        outputs=["dataframe"],
    ).launch()
