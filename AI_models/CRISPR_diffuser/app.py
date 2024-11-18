import gradio as gr
from diffusers import DiffusionPipeline
import torch
import pandas as pd
from tempfile import mkstemp
from PIL import Image
from .inference import data_collector_inference
from ..config import get_config, get_logger
from .scheduler import scheduler
from .dynamics import draw

args = get_config(config_file="config_CRISPR_diffuser.ini")
logger = get_logger(args)

@torch.no_grad()
def app(data_name=args.data_name):
    logger.info("get scheduler")
    noise_scheduler = scheduler()

    logger.info("setup pipeline")
    pipe = DiffusionPipeline.from_pretrained(f"{args.owner}/{data_name}_CRISPR_diffuser", trust_remote_code=True, custom_pipeline=f"{args.owner}/{data_name}_CRISPR_diffuser")
    pipe.unet.to(args.device)

    def gradio_fn(ref, cut):
        batch = data_collector_inference(
            [{
                "ref": ref,
                "cut": cut
            }],
            noise_scheduler,
            pipe.stationary_sampler1,
            pipe.stationary_sampler2
        )
        x1ts, x2ts, _ = pipe(batch, batch_size=args.batch_size, record_path=True)
        filename = mkstemp()[1]
        draw(x1ts, x2ts, filename, interval=120, pad=5)
        sequence = [ref[:x10] + ref[-args.ref2len+x20:] for x10, x20 in zip(x1ts[0], x2ts[0])]

        return (
            pd.DataFrame(
                {
                    "sequence": sequence
                }
            ),
            Image.open(filename)
        )

    gr.Interface(
        fn=gradio_fn,
        inputs=["text", "number"],
        outputs=["dataframe", "image"],
    ).launch()
