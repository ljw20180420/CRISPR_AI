import gradio as gr
from diffusers import DiffusionPipeline
import torch
import pandas as pd
from .inference import data_collector_inference
from ..config import args, logger

@torch.no_grad()
def app(data_name=args.data_name):
    logger.info("get scheduler")
    if args.noise_scheduler == "linear":
        from .scheduler import CRISPRDiffuserLinearScheduler
        noise_scheduler = CRISPRDiffuserLinearScheduler(
            num_train_timesteps = args.noise_timesteps
        )
    elif args.noise_scheduler == "cosine":
        from .scheduler import CRISPRDiffuserCosineScheduler
        noise_scheduler = CRISPRDiffuserCosineScheduler(
            num_train_timesteps = args.noise_timesteps,
            cosine_factor = args.cosine_factor
        )
    elif args.noise_scheduler == "exp":
        from .scheduler import CRISPRDiffuserExpScheduler
        noise_scheduler = CRISPRDiffuserExpScheduler(
            num_train_timesteps = args.noise_timesteps,
            exp_scale = args.exp_scale,
            exp_base = args.exp_base
        )
    elif args.noise_scheduler == "uniform":
        from .scheduler import CRISPRDiffuserUniformScheduler
        noise_scheduler = CRISPRDiffuserUniformScheduler(
            num_train_timesteps = args.noise_timesteps,
            uniform_scale = args.uniform_scale
        )

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
        x1t, x2t = pipe(batch, batch_size=1, record_path=False)
        return pd.DataFrame(
            {
                "x1": x1t.tolist(),
                "x2": x2t.tolist()
            }
        )

    gr.Interface(
        fn=gradio_fn,
        inputs=["text", "number"],
        outputs=["dataframe"],
    ).launch()
