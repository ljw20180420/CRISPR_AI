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
        count_normalize: float = 1000.,
        channels: List = [11, 32, 64, 96, 64, 32, 1],
        MCMC_corrector_factor: List = [1., 0., 0.001],
        ref1len: int = 127,
        ref2len: int = 127,
        seed: int = 63036, # random seed for intialization
        **kwargs,
    ):
        self.count_normalize = count_normalize
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
        self.register_buffer("stationary_sampler1_probs", F.normalize(torch.ones(config.ref1len + 1), p=1.0, dim=0))
        self.register_buffer("stationary_sampler2_probs", F.normalize(torch.ones(config.ref2len + 1), p=1.0, dim=0))
        # time
        self.time_emb = nn.Sequential(
            nn.Linear(in_features=self.config.channels[1], out_features=4 * self.config.channels[1]),
            nn.SiLU(),
            nn.Linear(in_features=4 * self.config.channels[1], out_features=4 * self.config.channels[1])
        )
        # down blocks
        self.down_time_embs = nn.ModuleList([])
        self.down_first_convs = nn.ModuleList([])
        self.down_second_convs = nn.ModuleList([])
        self.down_samples = nn.ModuleList([])
        for i in range((len(self.config.channels) - 1) // 2 - 1):
            self.down_first_convs.append(nn.Sequential(
                nn.Conv2d(in_channels=self.config.channels[i], out_channels=self.config.channels[i + 1], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=self.config.channels[i + 1]),
                nn.SiLU(inplace=True)
            ))
            self.down_second_convs.append(nn.Sequential(
                nn.Conv2d(in_channels=self.config.channels[i + 1], out_channels=self.config.channels[i + 1], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=self.config.channels[i + 1]),
                nn.SiLU(inplace=True),
            ))
            self.down_time_embs.append(nn.Sequential(
                nn.Linear(in_features=4 * self.config.channels[1], out_features=self.config.channels[i + 1]),
                nn.SiLU()
            ))
            self.down_samples.append(
                nn.MaxPool2d(kernel_size=2) # nn.AvgPool2d(kernel_size=2), nn.Conv2d(channels[i + 1], channels[i + 1], kernel_size=2, stride=2)
            )
        # mid block
        i = (len(self.config.channels) - 1) // 2 - 1
        self.mid_first_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.config.channels[i], out_channels=self.config.channels[i + 1], kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=self.config.channels[i + 1]),
            nn.SiLU(inplace=True)
        )
        self.mid_second_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.config.channels[i + 1], out_channels=self.config.channels[i + 1], kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=self.config.channels[i + 1]),
            nn.SiLU(inplace=True),
        )
        self.mid_time_emb = nn.Sequential(
            nn.Linear(in_features=4 * self.config.channels[1], out_features=self.config.channels[i + 1]),
            nn.SiLU()
        )
        # up blocks
        self.up_samples = nn.ModuleList([])
        self.up_time_embs = nn.ModuleList([])
        self.up_first_convs = nn.ModuleList([])
        self.up_second_convs = nn.ModuleList([])
        for i in range((len(self.config.channels) - 1) // 2, len(self.config.channels) - 2):
            self.up_samples.append(
                nn.ConvTranspose2d(in_channels=self.config.channels[i], out_channels=self.config.channels[i + 1], kernel_size=2, stride=2)
            )
            self.up_time_embs.append(nn.Sequential(
                nn.Linear(in_features=4 * self.config.channels[1], out_features=self.config.channels[i + 1]),
                nn.SiLU()
            ))
            self.up_first_convs.append(nn.Sequential(
                nn.Conv2d(in_channels=self.config.channels[i + 1]+self.config.channels[len(self.config.channels) - i - 2], out_channels=self.config.channels[i + 1], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=self.config.channels[i + 1]),
                nn.SiLU(inplace=True)
            ))
            self.up_second_convs.append(nn.Sequential(
                nn.Conv2d(in_channels=self.config.channels[i + 1], out_channels=self.config.channels[i + 1], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=self.config.channels[i + 1]),
                nn.SiLU(inplace=True)
            ))
        self.out_cov = nn.Conv2d(in_channels=self.config.channels[-2], out_channels=self.config.channels[-1], kernel_size=1)

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
        t_emb = get_timestep_embedding(t, embedding_dim=self.config.channels[1], flip_sin_to_cos=True, downscale_freq_shift=0)
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
        def get_q_rkm_d(stationary_sampler_probs, xt):
            xt_one_hot = F.one_hot(xt, len(stationary_sampler_probs))
            q_rkm_d = alpha_t[:, None] * xt_one_hot + ((1 - alpha_t) * stationary_sampler_probs[xt])[:, None]
            return q_rkm_d
        
        def get_g_theta_d(stationary_sampler_probs, xt, dim, p_theta_0):
            auxilary_term = 1 + (1 / alpha_t - 1) * stationary_sampler_probs[xt]
            xt_one_hot = F.one_hot(xt, len(stationary_sampler_probs))
            p_theta_d_0 = p_theta_0.sum(dim=dim)
            g_theta_d = (
                (1 - p_theta_d_0[torch.arange(p_theta_d_0.shape[0]), xt] / auxilary_term)[:, None] * stationary_sampler_probs +
                (alpha_t / (1 - alpha_t))[:, None] * p_theta_d_0
            ) * (1 - xt_one_hot) / stationary_sampler_probs[xt][:, None] + xt_one_hot
            return g_theta_d

        alpha_t = torch.e ** (-t)
        batch_size = p_theta_0_logit.shape[0]
        p_theta_0 = F.softmax(
            p_theta_0_logit.view(batch_size, -1),
            dim = 1
        ).view(batch_size, len(self.stationary_sampler2_probs), len(self.stationary_sampler1_probs))
        log_p_theta_0 = F.log_softmax(
            p_theta_0_logit.view(batch_size, -1),
            dim = 1
        ).view(batch_size, len(self.stationary_sampler2_probs), len(self.stationary_sampler1_probs))

        g_theta_1_t = get_g_theta_d(self.stationary_sampler1_probs, x1t, 1, p_theta_0)
        g_theta_2_t = get_g_theta_d(self.stationary_sampler2_probs, x2t, 2, p_theta_0)

        q_rkm_1 = get_q_rkm_d(self.stationary_sampler1_probs, x1t)
        q_rkm_2 = get_q_rkm_d(self.stationary_sampler2_probs, x2t)
        q_0_give_t = F.normalize(
            (observation * q_rkm_1[:, None, :] * q_rkm_2[:, :, None]).view(batch_size, -1),
            p=1.0, dim=1
        ).view(batch_size, len(self.stationary_sampler2_probs), len(self.stationary_sampler1_probs))

        g_1_t = get_g_theta_d(self.stationary_sampler1_probs, x1t, 1, q_0_give_t)
        g_2_t = get_g_theta_d(self.stationary_sampler2_probs, x2t, 2, q_0_give_t)

        common_negative_ELBO = (
            self.stationary_sampler1_probs[x1t] * g_theta_1_t.sum(dim = 1) +
            self.stationary_sampler2_probs[x2t] * g_theta_2_t.sum(dim = 1)
        )

        log_g_theta_1_t = g_theta_1_t.log().clamp(-1000, torch.inf)
        log_g_theta_2_t = g_theta_2_t.log().clamp(-1000, torch.inf)

        forward_negative_ELBO = common_negative_ELBO + (
            torch.inner(self.stationary_sampler1_probs, log_g_theta_1_t) +
            torch.inner(self.stationary_sampler2_probs, log_g_theta_2_t)
        )

        reverse_negative_ELBO = common_negative_ELBO - (
            (g_1_t * log_g_theta_1_t).sum(dim=1) +
            (g_2_t * log_g_theta_2_t).sum(dim=1)
        )

        MCMC_corrector = - (log_p_theta_0.view(batch_size, -1) * q_0_give_t.view(batch_size, -1)).sum(dim=1)

        return ( 
            observation.sum(dim=(1, 2)) / self.config.count_normalize * (
                self.config.MCMC_corrector_factor[0] * forward_negative_ELBO +
                self.config.MCMC_corrector_factor[1] * reverse_negative_ELBO +
                self.config.MCMC_corrector_factor[2] * MCMC_corrector
            )
        ).sum()
    
    # transformers.modeling_utils.ModuleUtilsMixin.floating_point_ops cannot handle nested input_dict, override it to avoid the error
    def floating_point_ops(self, input_dict: Dict[str, Union[torch.Tensor, Any]]):
        return 0
