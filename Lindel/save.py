import os
import torch
from huggingface_hub import create_repo, upload_folder
from config import args

def save_model(model, ymd, hms, epoch):
    os.makedirs(args.data_file.parent / "Lindel" / "output" / "models" / ymd, exist_ok=True)
    torch.save(model.state_dict(), args.data_file.parent / "Lindel" / "output" / "models" / ymd / f"{hms}_epoch{epoch}.pth")

def push_model(ymd, hms, epoch):
    repo_id = create_repo(
        repo_id="ljw20180420/Lindel", exist_ok=True
    ).repo_id
    upload_folder(
        repo_id=repo_id,
        folder_path=(args.dafa_file.parent / "Lindel" / "output" / "models"),
        commit_message=f"{ymd} {hms} epoch{epoch}",
        ignore_patterns=["step_*", "epoch_*"],
    )
        
