import gradio as gr
from diffusers import DiffusionPipeline
import torch


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
    pipe.core_model.to(device)

    def gradio_fn(ref, cut):
        return pipe.inference(examples=[{"ref": ref, "cut": cut}])

    gr.Interface(
        fn=gradio_fn,
        inputs=["text", "number"],
        outputs=["dataframe"],
    ).launch()
