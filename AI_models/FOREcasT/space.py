from huggingface_hub import HfApi
from huggingface_hub import HfFileSystem
from .model import FOREcasTConfig
from ..config import args

def space(data_name=args.data_name):
    api = HfApi()
    fs = HfFileSystem()
    api.create_repo(
        repo_id=f"{args.owner}/{data_name}_{FOREcasTConfig.model_type}",
        repo_type="space",
        exist_ok=True,
        space_sdk="gradio"
    )

    api.upload_file(
        repo_id=f"{args.owner}/{data_name}_{FOREcasTConfig.model_type}",
        repo_type="space",
        path_or_fileobj="AI_models/FOREcasT/app.py",
        path_in_repo="AI_models/FOREcasT/app.py"
    )

    api.upload_file(
        repo_id=f"{args.owner}/{data_name}_{FOREcasTConfig.model_type}",
        repo_type="space",
        path_or_fileobj="AI_models/config.py",
        path_in_repo="AI_models/config.py"
    )
    api.upload_file(
        repo_id=f"{args.owner}/{data_name}_{FOREcasTConfig.model_type}",
        repo_type="space",
        path_or_fileobj="config.ini",
        path_in_repo="config.ini"
    )

    api.upload_file(
        repo_id=f"{args.owner}/{data_name}_{FOREcasTConfig.model_type}",
        repo_type="space",
        path_or_fileobj="AI_models/FOREcasT/inference.py",
        path_in_repo="AI_models/FOREcasT/inference.py"
    )
    api.upload_file(
        repo_id=f"{args.owner}/{data_name}_{FOREcasTConfig.model_type}",
        repo_type="space",
        path_or_fileobj="AI_models/dataset/CRISPR_data.py",
        path_in_repo="AI_models/dataset/CRISPR_data.py"
    )
    api.upload_file(
        repo_id=f"{args.owner}/{data_name}_{FOREcasTConfig.model_type}",
        repo_type="space",
        path_or_fileobj="AI_models/FOREcasT/load_data.py",
        path_in_repo="AI_models/FOREcasT/load_data.py"
    )
    
    with fs.open(f"spaces/{args.owner}/{data_name}_{FOREcasTConfig.model_type}/app.py", "w") as fd:
        fd.write(f"from AI_models.FOREcasT.app import app\n")
        fd.write(f'''app(data_name="{data_name}")\n''')
    with fs.open(f"spaces/{args.owner}/{data_name}_{FOREcasTConfig.model_type}/requirements.txt", "w") as fd:
        fd.write(
            "\n".join([
                "accelerate",
                "ConfigArgParse",
                "transformers",
                "diffusers",
                "torch"
            ])
        )