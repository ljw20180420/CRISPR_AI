import os
import torch
from huggingface_hub import create_repo, upload_folder
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from config import args
import matplotlib

matplotlib.use('Agg')

def save_model(model, ymd, hms, epoch):
    os.makedirs(args.data_file.parent / "CRISPRdiffuser" / "output" / "models" / ymd, exist_ok=True)
    torch.save(model.state_dict(), args.data_file.parent / "CRISPRdiffuser" / "output" / "models" / ymd / f"{hms}_epoch{epoch}.pth")

def push_model(ymd, hms, epoch):
    repo_id = create_repo(
        repo_id="ljw20180420/CRISPRdiffuser", exist_ok=True
    ).repo_id
    upload_folder(
        repo_id=repo_id,
        folder_path=(args.dafa_file.parent / "CRISPRdiffuser" / "output" / "models"),
        commit_message=f"{ymd} {hms} epoch{epoch}",
        ignore_patterns=["step_*", "epoch_*"],
    )

def save_heatmap(matrix, name):
    os.makedirs(name.parent, exist_ok=True)
    fig, ax = plt.subplots()
    im = ax.imshow(matrix.cpu().numpy() ** args.display_scale_facter, vmin=0, cmap=LinearSegmentedColormap.from_list("", [(0, "white"), (1, "red")]))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    fig.savefig(name)
    plt.close(fig)
        
