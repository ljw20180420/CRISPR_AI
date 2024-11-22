import gradio as gr
from diffusers import DiffusionPipeline
import torch
import pandas as pd
from .inference import data_collector_inference
from ..config import get_config, get_logger

args = get_config(config_file="config_Lindel.ini")
logger = get_logger(args)

@torch.no_grad()
def app(data_name=args.data_name):
    logger.info("setup pipeline")
    pipe = DiffusionPipeline.from_pretrained(f"{args.owner}/{data_name}_Lindel", trust_remote_code=True, custom_pipeline=f"{args.owner}/{data_name}_Lindel")
    pipe.indel_model.to(args.device)
    pipe.ins_model.to(args.device)
    pipe.del_model.to(args.device)

    def gradio_fn(ref, cut):
        batch = data_collector_inference(
            [{
                "ref": ref,
                "cut": cut
            }]
        )
        del_proba, ins_proba, ins_base, ins_base_proba, dstart, dend, del_pos_proba = pipe(batch).values()

        df = pd.DataFrame(
            {
                "Category": ["del"] * len(dstart) + ["ins"] * len(ins_base),
                "Genotype position": dstart + [0] * len(ins_base),
                "Inserted Bases": [""] * len(dstart) + ins_base,
                "Length": [de - ds for ds, de in zip(dstart, dend)] + [len(ins) if ins != '>2' else '>2' for ins in ins_base],
                "Predicted frequency": (del_proba[0] * del_pos_proba[0]).tolist() + (ins_proba[0] * ins_base_proba[0]).tolist()
            }
        )
        df["sequence"] = [ref[:left+cut] + ins + ref[left+int(length)+cut:] for left, length, ins in zip(df["Genotype position"], df["Length"], df["Inserted Bases"]) if length != '>2'] + [""]

        return df.sort_values(by="Predicted frequency", ascending=False)

    gr.Interface(
        fn=gradio_fn,
        inputs=["text", "number"],
        outputs=["dataframe"],
    ).launch()
