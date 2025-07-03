import logging
import torch
from diffusers import DiffusionPipeline
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


@torch.no_grad()
def upload(
    preprocess: str,
    model: str,
    data_name: str,
    owner: str,
    logger: logging.Logger,
):
    logger.info("load pipeline")
    pipe = DiffusionPipeline.from_pretrained(
        f"{preprocess}/pipeline/{model}/{data_name}",
        custom_pipeline=f"{preprocess}/pipeline/{model}/{data_name}",
    )

    logger.info("push pipeline to hub")
    pipe.push_to_hub(f"{owner}/{preprocess}_{model}_{data_name}")
    from huggingface_hub import HfApi

    api = HfApi()
    while True:
        try:
            api.upload_file(
                repo_id=f"{owner}/{preprocess}_{model}_{data_name}",
                path_or_fileobj=f"{preprocess}/pipeline/{model}/{data_name}/pipeline.py",
                path_in_repo="pipeline.py",
            )
            for component in pipe.components.keys():
                api.upload_folder(
                    repo_id=f"{owner}/{preprocess}_{model}_{data_name}",
                    folder_path=f"{preprocess}/pipeline/{model}/{data_name}/{component}",
                    path_in_repo=component,
                    ignore_patterns=["_*", f"{PREFIX_CHECKPOINT_DIR}-*"],
                )

            break
        except Exception as err:
            print(err)
            print("retry")
