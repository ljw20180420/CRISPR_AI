from diffusers.models.embeddings import get_timestep_embedding
import torch
import torch.nn as nn
from torch.distributions import Categorical
from .config import args, ref1len, ref2len, device
from .load_data import train_dataloader

if args.forward_stationary_distribution == "uniform":
    # Categorical normalize probs automatically
    stationary_sampler1 = Categorical(probs=torch.ones(ref1len + 1, device=device))
    stationary_sampler2 = Categorical(probs=torch.ones(ref2len + 1, device=device))

for batch in train_dataloader:
    break

class Unet(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.channels = channels
        # time
        self.time_emb = nn.Sequential(
            nn.Linear(in_features=self.channels[1], out_features=4 * self.channels[1]),
            nn.SiLU(),
            nn.Linear(in_features=4 * self.channels[1], out_features=4 * self.channels[1])
        )
        # down blocks
        self.down_time_embs = nn.ModuleList([])
        self.down_first_convs = nn.ModuleList([])
        self.down_second_convs = nn.ModuleList([])
        self.down_samples = nn.ModuleList([])
        for i in range((len(channels) - 1) // 2 - 1):
            self.down_first_convs.append(nn.Sequential(
                nn.Conv2d(in_channels=channels[i], out_channels=channels[i + 1], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=channels[i + 1]),
                nn.SiLU(inplace=True)
            ))
            self.down_second_convs.append(nn.Sequential(
                nn.Conv2d(in_channels=channels[i + 1], out_channels=channels[i + 1], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=channels[i + 1]),
                nn.SiLU(inplace=True),
            ))
            self.down_time_embs.append(nn.Sequential(
                nn.Linear(in_features=4 * self.channels[1], out_features=channels[i + 1]),
                nn.SiLU()
            ))
            self.down_samples.append(
                nn.MaxPool2d(kernel_size=2) # nn.AvgPool2d(kernel_size=2), nn.Conv2d(channels[i + 1], channels[i + 1], kernel_size=2, stride=2)
            )
        # mid block
        i = (len(channels) - 1) // 2 - 1
        self.mid_first_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels[i], out_channels=channels[i + 1], kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=channels[i + 1]),
            nn.SiLU(inplace=True)
        )
        self.mid_second_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels[i + 1], out_channels=channels[i + 1], kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=channels[i + 1]),
            nn.SiLU(inplace=True),
        )
        self.mid_time_emb = nn.Sequential(
            nn.Linear(in_features=4 * self.channels[1], out_features=channels[i + 1]),
            nn.SiLU()
        )
        # up blocks
        self.up_samples = nn.ModuleList([])
        self.up_time_embs = nn.ModuleList([])
        self.up_first_convs = nn.ModuleList([])
        self.up_second_convs = nn.ModuleList([])
        for i in range((len(channels) - 1) // 2, len(channels) - 2):
            self.up_samples.append(
                nn.ConvTranspose2d(in_channels=channels[i], out_channels=channels[i + 1], kernel_size=2, stride=2)
            )
            self.up_time_embs.append(nn.Sequential(
                nn.Linear(in_features=4 * self.channels[1], out_features=channels[i + 1]),
                nn.SiLU()
            ))
            self.up_first_convs.append(nn.Sequential(
                nn.Conv2d(in_channels=channels[i + 1]+channels[len(channels) - i - 2], out_channels=channels[i + 1], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=channels[i + 1]),
                nn.SiLU(inplace=True)
            ))
            self.up_second_convs.append(nn.Sequential(
                nn.Conv2d(in_channels=channels[i + 1], out_channels=channels[i + 1], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=channels[i + 1]),
                nn.SiLU(inplace=True)
            ))
        self.out_cov = nn.Conv2d(in_channels=channels[-2], out_channels=channels[-1], kernel_size=1)
        

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        t_emb = get_timestep_embedding(t, embedding_dim=self.channels[1], flip_sin_to_cos=True, downscale_freq_shift=0)
        t_emb = self.time_emb(t_emb)
        down_xs = []
        for i in range(len(self.down_first_convs)):
            down_xs.append(
                self.down_second_convs[i](self.down_first_convs[i](x) + self.down_time_embs[i](t_emb)[:, :, None, None])
            )
            x = self.down_samples[i](down_xs[-1])
        x = self.mid_second_conv(self.mid_first_conv(x) + self.mid_time_emb(t_emb)[:, :, None, None])
        for i in range(len(self.up_first_convs)):
            x = self.up_second_convs[i](self.up_first_convs[i](torch.cat((down_xs.pop(), self.up_samples[i](x)), dim=1)) + self.up_time_embs[i](t_emb)[:, :, None, None])
        return self.out_cov(x)
            





model = Unet([batch["condition"].shape[1] + 1] + args.unet_channels + [1]).to(device)
