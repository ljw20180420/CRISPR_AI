import gradio as gr
from diffusers import DiffusionPipeline
import torch
import pandas as pd


@torch.no_grad()
def app(
    preprocess: str,
    model_name: str,
    data_name: str,
    owner: str,
    device: str,
) -> None:
    pipe = DiffusionPipeline.from_pretrained(
        f"{owner}/{preprocess}_{model_name}_{data_name}",
        trust_remote_code=True,
        custom_pipeline=f"{owner}/{preprocess}_{model_name}_{data_name}",
    )
    if hasattr(pipe, "core_model"):
        pipe.core_model.to(device)
    if hasattr(pipe, "auxilary_model"):
        pipe.auxilary_model.load_auxilary(
            f"{owner}/{preprocess}_{model_name}_{data_name}/auxilary_model/auxilary.pkl"
        )

    if preprocess == "DeepHF":

        def gradio_fn(ref: str, cut: int, scaffold: str) -> pd.DataFrame:
            return pipe.inference(
                examples=[{"ref": ref, "cut": cut, "scaffold": scaffold}]
            )

        gr.Interface(
            fn=gradio_fn,
            inputs=["text", "number", "text"],
            outputs=["dataframe"],
        ).launch()

    else:

        def gradio_fn(ref: str, cut: int) -> pd.DataFrame:
            return pipe.inference(examples=[{"ref": ref, "cut": cut}])

        gr.Interface(
            fn=gradio_fn,
            inputs=["text", "number"],
            outputs=["dataframe"],
        ).launch()
