from diffusers.models.embeddings import get_timestep_embedding
from transformers import PretrainedConfig, PreTrainedModel
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Dict, Union, Any, List

class CRISPRDiffuserConfig(PretrainedConfig):
    model_type = "CRISPR_diffuser"
    label_names = ["observation"]
    main_input_name = "x1t_x2t_t"

    def __init__(
        self,
        channels: List = [11, 32, 64, 96, 64, 32, 1],
        MCMC_corrector_factor: float = 0.001,
        ref1len: int = 127,
        ref2len: int = 127,
        seed: int = 63036, # random seed for intialization
        **kwargs,
    ):
        self.channels = channels
        self.MCMC_corrector_factor = MCMC_corrector_factor
        self.ref1len = ref1len
        self.ref2len = ref2len
        self.seed = seed
        super().__init__(**kwargs)

class CRISPRDiffuserModel(PreTrainedModel):
    config_class = CRISPRDiffuserConfig

    def __init__(self, config):
        super().__init__(config)
        # In more recent versions of PyTorch, you no longer need to explicitly register_parameter, it's enough to set a member of your nn.Module with nn.Parameter to "notify" pytorch that this variable should be treated as a trainable parameter (https://stackoverflow.com/questions/59234238/how-to-add-parameters-in-module-class-in-pytorch-custom-model).
        self.main_input_name = config.main_input_name
        # record loss inside model to stop training in callbacks
        self.loss = None
        self.generator = torch.Generator().manual_seed(config.seed)
        self.channels = config.channels
        self.MCMC_corrector_factor = config.MCMC_corrector_factor
        self.register_buffer("stationary_sampler1_probs", F.normalize(torch.ones(config.ref1len + 1), p=1.0, dim=0))
        self.register_buffer("stationary_sampler2_probs", F.normalize(torch.ones(config.ref2len + 1), p=1.0, dim=0))
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

    def forward(self, x1t_x2t_t: dict, condition: torch.Tensor, observation: torch.Tensor | None = None):
        x1t, x2t, t = x1t_x2t_t.values()
        batch_size = condition.shape[0]
        x = torch.cat((
            (
                F.one_hot(x1t, num_classes=len(self.stationary_sampler1_probs)).view(batch_size, 1, -1) *
                F.one_hot(x2t, num_classes=len(self.stationary_sampler2_probs)).view(batch_size, -1, 1)
            )[:, None, :, :],
            condition
        ), dim = 1)
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
            self.loss = self.continuous_time_loss_function(x1t, x2t, t, p_theta_0_logit, observation)
            return {
                "p_theta_0_logit": p_theta_0_logit,
                "loss": self.loss
            }
        return {
            "p_theta_0_logit": p_theta_0_logit
        }

    def continuous_time_loss_function(self, x1t: torch.Tensor, x2t: torch.Tensor, t: torch.Tensor, p_theta_0_logit: torch.Tensor, observation: torch.Tensor):
        def get_g_theta_d_and_q_rkm(stationary_sampler_probs, xt, dim, p_theta_0):
            auxilary_term = 1 + (1 / alpha_t - 1) * stationary_sampler_probs[xt]
            xt_one_hot = F.one_hot(xt, len(stationary_sampler_probs))
            p_theta_d_0 = p_theta_0.sum(dim=dim)
            g_theta_d = (
                (1 - p_theta_d_0[torch.arange(p_theta_d_0.shape[0]), xt] / auxilary_term)[:, None] * stationary_sampler_probs +
                (alpha_t / (1 - alpha_t))[:, None] * p_theta_d_0
            ) * (1 - xt_one_hot) / stationary_sampler_probs[xt][:, None] + xt_one_hot
            q_rkm = alpha_t[:, None] * xt_one_hot + ((1 - alpha_t) * stationary_sampler_probs[xt])[:, None]
            return g_theta_d, q_rkm

        alpha_t = torch.e ** (-t)
        batch_size = p_theta_0_logit.shape[0]
        p_theta_0 = F.softmax(
            p_theta_0_logit.view(batch_size, -1),
            dim = 1
        ).view(batch_size, len(self.stationary_sampler2_probs), len(self.stationary_sampler1_probs))

        g_theta_1_t, q_rkm_1 = get_g_theta_d_and_q_rkm(self.stationary_sampler1_probs, x1t, 1, p_theta_0)
        g_theta_2_t, q_rkm_2 = get_g_theta_d_and_q_rkm(self.stationary_sampler2_probs, x2t, 2, p_theta_0)
        q_0_give_t = F.normalize(
            (observation * q_rkm_1[:, None, :] * q_rkm_2[:, :, None]).view(batch_size, -1),
            p=1.0, dim=1
        )

        return (
            self.stationary_sampler1_probs[x1t] * g_theta_1_t.sum(dim = 1) +
            torch.inner(self.stationary_sampler1_probs, g_theta_1_t.log()) +
            self.stationary_sampler2_probs[x2t] * g_theta_2_t.sum(dim = 1) +
            torch.inner(self.stationary_sampler2_probs, g_theta_2_t.log()) -
            self.config.MCMC_corrector_factor * (p_theta_0.log().view(batch_size, -1) * q_0_give_t).sum(dim=1)
        ).sum()
    
    # transformers.modeling_utils.ModuleUtilsMixin.floating_point_ops cannot handle nested input_dict, override it to avoid the error
    def floating_point_ops(self, input_dict: Dict[str, Union[torch.Tensor, Any]]):
        return 0
