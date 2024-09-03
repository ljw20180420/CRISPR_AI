from captum.attr import IntegratedGradients
import torch
import torch.nn.functional as F
import os
import pathlib
import matplotlib.pyplot as plt
from CRISPR_diffuser.config import args, device, ref1len, ref2len
from CRISPR_diffuser.load_data import valid_dataloader
from CRISPR_diffuser.model import model, stationary_sampler1, stationary_sampler2
from CRISPR_diffuser.noise_scheduler import noise_scheduler

date = "2024-07-26"
name = "11:33:16_epoch4.pth"
model_state = pathlib.Path("/home/ljw/sdc1/SX/output/models") / date / name

model.load_state_dict(torch.load(model_state))
model.eval()
model.to(device)

for batch in valid_dataloader:
    break

x1t = stationary_sampler1.sample(torch.Size([1]))
x2t = stationary_sampler2.sample(torch.Size([1]))
t = noise_scheduler(torch.rand(1, device=device) * args.noise_timesteps)
x1t_one_hot = F.one_hot(x1t, num_classes=ref1len + 1)
x2t_one_hot = F.one_hot(x2t, num_classes=ref2len + 1)
values = x1t_one_hot.view(1, 1, -1) * x2t_one_hot.view(1, -1, 1)
input = torch.concat((
    values.unsqueeze(1),
    batch['condition']
), dim = 1)

vals, idxs = torch.topk(batch['observation'].flatten(), 10)
ref2idxs = idxs // (ref1len + 1)
ref1idxs = idxs % (ref1len + 1)

ig = IntegratedGradients(model)

mh_matrices = []
cut_one_hots = []
ref1_mean = []
ref2_mean = []
spos = (input.shape[1] - 3) // 2 + 3
for ref2idx in ref2idxs:
    for ref1idx in ref1idxs:
        attr = ig.attribute((input, t), target=(1, ref2idx, ref1idx))[0]
        mh_matrices.append(attr[0, 1])
        cut_one_hots.append(attr[0, 2])
        ref1_mean.append(attr[0, 3:spos].mean(dim=0))
        ref2_mean.append(attr[0, spos:].mean(dim=0))

def save_target(ref1idxs, ref2idxs):
    save_dir = pathlib.Path("interprets/") / date / name
    os.makedirs(save_dir, exist_ok=True)
    max_ob = batch['observation'].max()
    for i in range(len(ref1idxs)):
        # save target figure
        fig, ax = plt.subplots()
        ax.imshow(batch['observation'], cmap="gray", vmin=0, vmax=max_ob, extent=(0, ref1len + 1, ref2len + 1, 0))
        ax.scatter(ref1idxs[i], ref2idxs[i], c="red")
        ax.colorbar()
        fig.savefig(save_dir / f"target{i}.png")

def save_attr(attrs, attr_name):
    save_dir = pathlib.Path("interprets/") / date / name
    os.makedirs(save_dir, exist_ok=True)
    vmax = max([attr.max() for attr in attrs])
    vmin = max([attr.min() for attr in attrs])
    for i in range(len(ref1idxs)):
        # save target figure
        fig, ax = plt.subplots()
        ax.imshow(attrs[i], cmap="gray", vmin=min, vmax=max, extent=(0, ref1len + 1, ref2len + 1, 0))
        ax.colorbar()
        fig.savefig(save_dir / f"{attr_name}{i}.png")

save_target(ref1idxs, ref2idxs)
save_attr(mh_matrices, "mh_matrices")
save_attr(cut_one_hots, "cut_one_hots")
save_attr(ref1_mean, "ref1_mean")
save_attr(ref2_mean, "ref2_mean")
