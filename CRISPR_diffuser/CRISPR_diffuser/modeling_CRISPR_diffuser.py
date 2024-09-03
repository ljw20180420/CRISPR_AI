from diffusers.models.embeddings import get_timestep_embedding
from transformers import PreTrainedModel
from .configuration_CRISPR_diffuser import CRISPRDiffuserConfig
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

class CRISPRDiffuserModel(PreTrainedModel):
    config_class = CRISPRDiffuserConfig

    def __init__(self, config):
        super().__init__(config)
        # In more recent versions of PyTorch, you no longer need to explicitly register_parameter, it's enough to set a member of your nn.Module with nn.Parameter to "notify" pytorch that this variable should be treated as a trainable parameter (https://stackoverflow.com/questions/59234238/how-to-add-parameters-in-module-class-in-pytorch-custom-model).
        self.generator = torch.Generator().manual_seed(config.seed)
        self.channels = config.channels
        self.MCMC_corrector_factor = config.MCMC_corrector_factor
        self.stationary_sampler1 = Categorical(probs=torch.ones(config.ref1len + 1))
        self.stationary_sampler2 = Categorical(probs=torch.ones(config.ref2len + 1))
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
        for i in range((len(self.channels) - 1) // 2 - 1):
            self.down_first_convs.append(nn.Sequential(
                nn.Conv2d(in_channels=self.channels[i], out_channels=self.channels[i + 1], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=self.channels[i + 1]),
                nn.SiLU(inplace=True)
            ))
            self.down_second_convs.append(nn.Sequential(
                nn.Conv2d(in_channels=self.channels[i + 1], out_channels=self.channels[i + 1], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=self.channels[i + 1]),
                nn.SiLU(inplace=True),
            ))
            self.down_time_embs.append(nn.Sequential(
                nn.Linear(in_features=4 * self.channels[1], out_features=self.channels[i + 1]),
                nn.SiLU()
            ))
            self.down_samples.append(
                nn.MaxPool2d(kernel_size=2) # nn.AvgPool2d(kernel_size=2), nn.Conv2d(channels[i + 1], channels[i + 1], kernel_size=2, stride=2)
            )
        # mid block
        i = (len(self.channels) - 1) // 2 - 1
        self.mid_first_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.channels[i], out_channels=self.channels[i + 1], kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=self.channels[i + 1]),
            nn.SiLU(inplace=True)
        )
        self.mid_second_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.channels[i + 1], out_channels=self.channels[i + 1], kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=self.channels[i + 1]),
            nn.SiLU(inplace=True),
        )
        self.mid_time_emb = nn.Sequential(
            nn.Linear(in_features=4 * self.channels[1], out_features=self.channels[i + 1]),
            nn.SiLU()
        )
        # up blocks
        self.up_samples = nn.ModuleList([])
        self.up_time_embs = nn.ModuleList([])
        self.up_first_convs = nn.ModuleList([])
        self.up_second_convs = nn.ModuleList([])
        for i in range((len(self.channels) - 1) // 2, len(self.channels) - 2):
            self.up_samples.append(
                nn.ConvTranspose2d(in_channels=self.channels[i], out_channels=self.channels[i + 1], kernel_size=2, stride=2)
            )
            self.up_time_embs.append(nn.Sequential(
                nn.Linear(in_features=4 * self.channels[1], out_features=self.channels[i + 1]),
                nn.SiLU()
            ))
            self.up_first_convs.append(nn.Sequential(
                nn.Conv2d(in_channels=self.channels[i + 1]+self.channels[len(self.channels) - i - 2], out_channels=self.channels[i + 1], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=self.channels[i + 1]),
                nn.SiLU(inplace=True)
            ))
            self.up_second_convs.append(nn.Sequential(
                nn.Conv2d(in_channels=self.channels[i + 1], out_channels=self.channels[i + 1], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=self.channels[i + 1]),
                nn.SiLU(inplace=True)
            ))
        self.out_cov = nn.Conv2d(in_channels=self.channels[-2], out_channels=self.channels[-1], kernel_size=1)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1, generator=self.generator)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=1, generator=self.generator)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0, std=1, generator=self.generator)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x1t: torch.Tensor, x2t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor, observation: torch.Tensor | None = None):
        batch_size = observation.shape[0]
        x = torch.cat((
            (
                F.one_hot(x1t, num_classes=self.stationary_sampler1._num_events).view(batch_size, 1, -1) *
                F.one_hot(x2t, num_classes=self.stationary_sampler2._num_events).view(batch_size, -1, 1)
            )[:, None, :, :],
            condition
        ), dim = 1),
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
        p_theta_0_logit = self.out_cov(x)
        if observation is not None:
            return {
                "p_theta_0_logit": p_theta_0_logit,
                "loss": self.continuous_time_loss_function(x1t, x2t, t, p_theta_0_logit, observation)
            }
        return {
            "p_theta_0_logit": p_theta_0_logit
        }

    def continuous_time_loss_function(self, x1t: torch.Tensor, x2t: torch.Tensor, t: torch.Tensor, p_theta_0_logit: torch.Tensor, observation: torch.Tensor):
        def get_g_theta_d_and_q_rkm(stationary_sampler, xt, dim, p_theta_0):
            auxilary_term = 1 + (torch.e ** (t) - 1) * stationary_sampler.probs[xt]
            xt_one_hot = F.one_hot(xt, stationary_sampler._num_events)
            p_theta_d_0 = p_theta_0.sum(dim=dim)
            g_theta_d = (
                (1 - p_theta_d_0[torch.arange(p_theta_d_0.shape[0]), xt] / auxilary_term)[:, None] * stationary_sampler.probs +
                (torch.e ** (-t) / (1 - torch.e ** (-t)))[:, None] * p_theta_d_0
            ) * (1 - xt_one_hot) / stationary_sampler.probs[xt][:, None] + xt_one_hot
            q_rkm = torch.e ** (-t)[:, None] * xt_one_hot + ((1 - torch.e ** (-t)) * stationary_sampler.probs[xt])[:, None]
            return g_theta_d, q_rkm

        batch_size = p_theta_0_logit.shape[0]
        p_theta_0 = F.softmax(
            p_theta_0_logit.view(batch_size, -1),
            dim = 1
        ).view(batch_size, self.stationary_sampler2._num_events, self.stationary_sampler1._num_events)

        g_theta_1_t, q_rkm_1 = get_g_theta_d_and_q_rkm(self.stationary_sampler1, x1t, 1, p_theta_0)
        g_theta_2_t, q_rkm_2 = get_g_theta_d_and_q_rkm(self.stationary_sampler2, x2t, 2, p_theta_0)
        q_0_give_t = (observation * q_rkm_1[:, None, :] * q_rkm_2[:, :, None]).view(batch_size, -1)
        q_0_give_t /= q_0_give_t.sum(dim=1, keepdim=True)

        return (
            self.stationary_sampler1.probs[x1t] * g_theta_1_t.sum(dim = 1) +
            torch.inner(self.stationary_sampler1.probs, g_theta_1_t.log()) +
            self.stationary_sampler2.probs[x2t] * g_theta_2_t.sum(dim = 1) +
            torch.inner(self.stationary_sampler2.probs, g_theta_2_t.log()) -
            self.config.MCMC_corrector_factor * (p_theta_0.log().view(batch_size, -1) * q_0_give_t).sum(dim=1)
        ).sum()