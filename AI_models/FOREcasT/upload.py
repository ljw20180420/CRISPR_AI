import logging
import torch
from diffusers import DiffusionPipeline
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from .model import FOREcasTConfig


@torch.no_grad()
def upload(
    data_name: str,
    owner: str,
    logger: logging.Logger,
):
    logger.info("load pipeline")
    pipe = DiffusionPipeline.from_pretrained(
        "FOREcasT/pipeline", custom_pipeline="FOREcasT/pipeline"
    )

    logger.info("push pipelien to hub")
    pipe.push_to_hub("%s/%s_%s" % (owner, data_name, FOREcasTConfig.model_type))
    from huggingface_hub import HfApi

    api = HfApi()
    while True:
        try:
            api.upload_file(
                repo_id="%s/%s_%s" % (owner, data_name, FOREcasTConfig.model_type),
                path_or_fileobj="FOREcasT/pipeline/pipeline.py",
                path_in_repo="pipeline.py",
            )
            for component in pipe.components.keys():
                api.upload_folder(
                    repo_id="%s/%s_%s" % (owner, data_name, FOREcasTConfig.model_type),
                    folder_path="FOREcasT/pipeline/%s" % component,
                    path_in_repo=component,
                    ignore_patterns=["_*", f"{PREFIX_CHECKPOINT_DIR}-*"],
                )

            break
        except Exception as err:
            print(err)
            print("retry")
