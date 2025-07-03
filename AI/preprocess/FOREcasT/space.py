from huggingface_hub import HfApi
from huggingface_hub import HfFileSystem
from .model import FOREcasTConfig


def space(
    data_name: str,
    ref1len: int,
    ref2len: int,
    owner: str,
    device: str,
):
    api = HfApi()
    fs = HfFileSystem()
    while True:
        try:
            api.create_repo(
                repo_id="%s/%s_%s" % (owner, data_name, FOREcasTConfig.model_type),
                repo_type="space",
                exist_ok=True,
                space_sdk="gradio",
            )

            api.upload_file(
                repo_id="%s/%s_%s" % (owner, data_name, FOREcasTConfig.model_type),
                repo_type="space",
                path_or_fileobj="FOREcasT/app.py",
                path_in_repo="FOREcasT/app.py",
            )

            api.upload_file(
                repo_id="%s/%s_%s" % (owner, data_name, FOREcasTConfig.model_type),
                repo_type="space",
                path_or_fileobj="FOREcasT/model.py",
                path_in_repo="FOREcasT/model.py",
            )
            api.upload_file(
                repo_id="%s/%s_%s" % (owner, data_name, FOREcasTConfig.model_type),
                repo_type="space",
                path_or_fileobj="FOREcasT/inference.py",
                path_in_repo="FOREcasT/inference.py",
            )
            api.upload_file(
                repo_id="%s/%s_%s" % (owner, data_name, FOREcasTConfig.model_type),
                repo_type="space",
                path_or_fileobj="FOREcasT/load_data.py",
                path_in_repo="FOREcasT/load_data.py",
            )

            with fs.open(
                "space/%s/%s_%s/app.py" % (owner, data_name, FOREcasTConfig.model_type),
                "w",
            ) as fd:
                fd.write("")
                fd.write(
                    """
from FOREcasT.app import app
app(
    data_name=%s,
    ref1len=%d,
    ref2len=%d,
    owner=%s,
    device=%s,
)
                    """
                    % (
                        data_name,
                        ref1len,
                        ref2len,
                        owner,
                        device,
                    )
                )
            with fs.open(
                "space/%s/%s_%s/requirements.txt"
                % (owner, data_name, FOREcasTConfig.model_type),
                "w",
            ) as fd:
                fd.write(
                    "\n".join(
                        [
                            "accelerate",
                            "transformers",
                            "diffusers",
                            "torch",
                        ]
                    )
                )
            break
        except Exception as err:
            print(err)
            print("retry")
