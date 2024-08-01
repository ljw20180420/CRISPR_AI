#!/usr/bin/env python

import torch
from torchviz import make_dot
from CRISPRdiffuser.model import model, stationary_sampler1, stationary_sampler2
from CRISPRdiffuser.load_data import valid_dataloader
from CRISPRdiffuser.noise_scheduler import noise_scheduler
from CRISPRdiffuser.config import args, ref1len, ref2len, device
import torch.nn.functional as F

for batch in valid_dataloader:
    break

model = model.to(device)
x1t = stationary_sampler1.sample(torch.Size([1]))
x2t = stationary_sampler2.sample(torch.Size([1]))
t = noise_scheduler(torch.tensor([args.noise_timesteps], device="cuda"))
s = noise_scheduler(torch.tensor([args.noise_timesteps - 1], device="cuda"))
x1t_one_hot = F.one_hot(x1t, num_classes=ref1len + 1)
x2t_one_hot = F.one_hot(x2t, num_classes=ref2len + 1)
values = x1t_one_hot.view(1, 1, -1) * x2t_one_hot.view(1, -1, 1)
p_theta_0_logit = model(
    torch.cat((
        values.unsqueeze(1),
        batch['condition'].to(values.device)
    ), dim = 1),
    t
)
dot = make_dot(p_theta_0_logit.mean(), params=dict(model.named_parameters()))
dot.render(filename="model.gv", format="svg")
