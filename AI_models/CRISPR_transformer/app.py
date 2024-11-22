import gradio as gr
from diffusers import DiffusionPipeline
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import io
import torch.nn.functional as F
from .inference import data_collector_inference
from ..config import get_config, get_logger

args = get_config(config_file="config_CRISPR_transformer.ini")
logger = get_logger(args)

@torch.no_grad()
def app(data_name=args.data_name):
    logger.info("setup pipeline")
    pipe = DiffusionPipeline.from_pretrained(f"{args.owner}/{data_name}_CRISPR_transformer", trust_remote_code=True, custom_pipeline=f"{args.owner}/{data_name}_CRISPR_transformer")
    pipe.CRISPR_transformer_model.to(args.device)

    def gradio_fn(ref, cut):
        batch = data_collector_inference(
            [{
                "ref": ref,
                "cut": cut
            }]
        )
        probability = F.softmax(pipe(batch)["logit"].flatten()).tolist()
        ref2start = np.arange(args.ref2len + 1).repeat(args.ref1len + 1)
        ref1end = list(range(args.ref1len + 1)) * (args.ref2len + 1)
        sequence = [ref[:re] + ref[len(ref)-args.ref2len+rs:] for rs, re in zip(ref2start, ref1end)]

        fig, ax = plt.subplots()
        im = ax.imshow(
            np.array(probability).reshape(args.ref2len + 1, args.ref1len + 1),
            cmap = LinearSegmentedColormap.from_list("white_to_black", [(0, "white"), (1, "black")])
        )
        fig.colorbar(im)
        bf = io.BytesIO()
        fig.savefig(bf)
        bf.seek(0)
        
        return (
            pd.DataFrame(
                {
                    "ref1end": ref1end,
                    "ref2start": ref2start,
                    "probability": probability,
                    "sequence": sequence
                }
            ).sort_values(by="probability", ascending=False),
            Image.open(bf)
        )

    gr.Interface(
        fn=gradio_fn,
        inputs=["text", "number"],
        outputs=["dataframe", "image"],
    ).launch()
