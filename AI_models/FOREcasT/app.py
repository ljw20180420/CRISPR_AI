import gradio as gr
from diffusers import DiffusionPipeline
import torch
import pandas as pd
from .inference import data_collector_inference
from ..config import args, logger

@torch.no_grad()
def app(data_name=args.data_name):
    logger.info("setup pipeline")
    pipe = DiffusionPipeline.from_pretrained(f"{args.owner}/{data_name}_FOREcasT", trust_remote_code=True, custom_pipeline=f"{args.owner}/{data_name}_FOREcasT", MAX_DEL_SIZE=args.FOREcasT_MAX_DEL_SIZE)
    pipe.FOREcasT_model.to(args.device)

    def gradio_fn(ref, cut):
        batch = data_collector_inference(
            [{
                "ref": ref,
                "cut": cut
            }]
        )
        proba, left, right, ins_seq = pipe(batch).values()
        del_num = sum([ins == '' for ins in ins_seq])
        ins_num = len(ins_seq) - del_num

        return pd.DataFrame(
            {
                "Category": ["del"] * del_num + ["ins"] * ins_num,
                "Genotype position": left.tolist(),
                "Inserted Bases": ins_seq,
                "Length": (right - left)[:del_num].tolist() + [len(ins) for ins in ins_seq[-ins_num:]],
                "Predicted frequency": proba.tolist()
            }
        )

    gr.Interface(
        fn=gradio_fn,
        inputs=["text", "number"],
        outputs=["dataframe"],
    ).launch()
