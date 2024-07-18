from huggingface_hub import create_repo, upload_folder
from .config import output_dir
import os

def save_model():
    os.makedirs(output_dir, exist_ok=True)
    pass

def push_model(epoch):
    repo_id = create_repo(
        repo_id="ljw20180420/CRISPRdiffuser", exist_ok=True
    ).repo_id
    upload_folder(
        repo_id=repo_id,
        folder_path=output_dir,
        commit_message=f"Epoch {epoch}",
        ignore_patterns=["step_*", "epoch_*"],
    )
        
