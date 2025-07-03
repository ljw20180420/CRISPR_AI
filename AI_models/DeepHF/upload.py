import logging
import torch
from diffusers import DiffusionPipeline
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


@torch.no_grad()
def upload_DeepHF(
    data_name: str,
    owner: str,
    logger: logging.Logger,
):
    logger.info("load pipeline")
    pipe = DiffusionPipeline.from_pretrained(
        f"DeepHF/pipeline/DeepHF/{data_name}",
        custom_pipeline=f"DeepHF/pipeline/DeepHF/{data_name}",
    )

    logger.info("push pipelien to hub")
    pipe.push_to_hub(f"{owner}/DeepHF_DeepHF_{data_name}")
    from huggingface_hub import HfApi

    api = HfApi()
    while True:
        try:
            api.upload_file(
                repo_id=f"{owner}/DeepHF_DeepHF_{data_name}",
                path_or_fileobj=f"DeepHF/pipeline/DeepHF/{data_name}/pipeline.py",
                path_in_repo="pipeline.py",
            )
            for component in pipe.components.keys():
                api.upload_folder(
                    repo_id=f"{owner}/DeepHF_DeepHF_{data_name}",
                    folder_path=f"DeepHF/pipeline/DeepHF/{data_name}/{component}",
                    path_in_repo=component,
                    ignore_patterns=["_*", f"{PREFIX_CHECKPOINT_DIR}-*"],
                )

            break
        except Exception as err:
            print(err)
            print("retry")
