import numpy as np
import pandas as pd
from diffusers.models.embeddings import get_timestep_embedding
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from transformers import PreTrainedModel, PretrainedConfig
from typing import Optional, Literal
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

# torch does not import opt_einsum as backend by default. import opt_einsum manually will enable it.
from torch.backends import opt_einsum
from einops import einsum, rearrange, repeat
from .data_collator import DataCollator
from common_ai.utils import MyGenerator


class CRIfuserConfig(PretrainedConfig):
    model_type = "CRIfuser"

    def __init__(
        self,
        ext1_up: int,
        ext1_down: int,
        ext2_up: int,
        ext2_down: int,
        max_micro_homology: int,
        loss_weights: dict[
            Literal[
                "double_sample_negative_ELBO",
                "importance_sample_negative_ELBO",
                "forward_negative_ELBO",
                "reverse_negative_ELBO",
                "sample_CE",
                "non_sample_CE",
            ],
            float,
        ],
        unet_channels: list[int],
        noise_scheduler: Literal["linear", "cosine", "exp", "uniform"],
        noise_timesteps: int,
        cosine_factor: float,
        exp_scale: float,
        exp_base: float,
        uniform_scale: float,
        **kwargs,
    ):
        """CRIfuser arguments.

        Args:
            ext1_up: upstream limit of the resection of the upstream end.
            ext1_down: downstream limit of the templated insertion of the upstream end.
            ext2_up: upstream limit of the templated insertion of the downstream end.
            ext2_down: downstream limit of the resection of the downstream end.
            max_micro_homology: clip micro-homology strength to (0, max_micro_homology).
            loss_weights: weights of loss functions.
            unet_channels: the channels of intermediate layers of unet.
            noise_scheduler: noise scheduler used for diffuser model.
            noise_timesteps: number of noise scheduler time steps.
            cosine_factor: parameter control cosine noise scheduler.
            exp_scale: scale factor of exponential noise scheduler.
            exp_base: base parameter of exponential noise scheduler.
            uniform_scale: scale parameter for uniform scheduler.
        """
        self.ext1_up = ext1_up
        self.ext1_down = ext1_down
        self.ext2_up = ext2_up
        self.ext2_down = ext2_down
        self.max_micro_homology = max_micro_homology
        self.loss_weights = loss_weights
        self.unet_channels = unet_channels
        self.noise_scheduler = noise_scheduler
        self.noise_timesteps = noise_timesteps
        self.cosine_factor = cosine_factor
        self.exp_scale = exp_scale
        self.exp_base = exp_base
        self.uniform_scale = uniform_scale
        super().__init__(**kwargs)


class CRIfuserModel(PreTrainedModel):
    config_class = CRIfuserConfig

    def __init__(self, config: CRIfuserConfig) -> None:
        super().__init__(config)
        self.data_collator = DataCollator(
            ext1_up=config.ext1_up,
            ext1_down=config.ext1_down,
            ext2_up=config.ext2_up,
            ext2_down=config.ext2_down,
            max_micro_homology=config.max_micro_homology,
        )
        self.stationary_sampler1 = Categorical(
            torch.ones(config.ext1_up + config.ext1_down + 1)
        )
        self.stationary_sampler2 = Categorical(
            torch.ones(config.ext2_up + config.ext2_down + 1)
        )
        # time
        self.time_emb = nn.Sequential(
            nn.Linear(
                in_features=config.unet_channels[0],
                out_features=4 * config.unet_channels[0],
            ),
            nn.SiLU(),
            nn.Linear(
                in_features=4 * config.unet_channels[0],
                out_features=4 * config.unet_channels[0],
            ),
        )
        # down blocks
        self.down_time_embs = nn.ModuleList([])
        self.down_first_convs = nn.ModuleList([])
        self.down_second_convs = nn.ModuleList([])
        self.down_samples = nn.ModuleList([])
        for i in range(len(config.unet_channels) // 2):
            self.down_first_convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=11 if i == 0 else config.unet_channels[i - 1],
                        out_channels=config.unet_channels[i],
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.BatchNorm2d(num_features=config.unet_channels[i]),
                    nn.SiLU(inplace=True),
                )
            )
            self.down_second_convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=config.unet_channels[i],
                        out_channels=config.unet_channels[i],
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.BatchNorm2d(num_features=config.unet_channels[i]),
                    nn.SiLU(inplace=True),
                )
            )
            self.down_time_embs.append(
                nn.Sequential(
                    nn.Linear(
                        in_features=4 * config.unet_channels[0],
                        out_features=config.unet_channels[i],
                    ),
                    nn.SiLU(),
                )
            )
            self.down_samples.append(
                nn.MaxPool2d(
                    kernel_size=2
                )  # nn.AvgPool2d(kernel_size=2), nn.Conv2d(config.unet_channels[i], config.unet_channels[i], kernel_size=2, stride=2)
            )
        # mid block
        i = len(config.unet_channels) // 2
        self.mid_first_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=config.unet_channels[i - 1],
                out_channels=config.unet_channels[i],
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=config.unet_channels[i]),
            nn.SiLU(inplace=True),
        )
        self.mid_second_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=config.unet_channels[i],
                out_channels=config.unet_channels[i],
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=config.unet_channels[i]),
            nn.SiLU(inplace=True),
        )
        self.mid_time_emb = nn.Sequential(
            nn.Linear(
                in_features=4 * config.unet_channels[0],
                out_features=config.unet_channels[i],
            ),
            nn.SiLU(),
        )
        # up blocks
        self.up_samples = nn.ModuleList([])
        self.up_time_embs = nn.ModuleList([])
        self.up_first_convs = nn.ModuleList([])
        self.up_second_convs = nn.ModuleList([])
        for i in range(len(config.unet_channels) // 2, len(config.unet_channels) - 1):
            self.up_samples.append(
                nn.ConvTranspose2d(
                    in_channels=config.unet_channels[i],
                    out_channels=config.unet_channels[i + 1],
                    kernel_size=2,
                    stride=2,
                )
            )
            self.up_time_embs.append(
                nn.Sequential(
                    nn.Linear(
                        in_features=4 * config.unet_channels[0],
                        out_features=config.unet_channels[i + 1],
                    ),
                    nn.SiLU(),
                )
            )
            self.up_first_convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=config.unet_channels[i + 1]
                        + config.unet_channels[len(config.unet_channels) - i - 2],
                        out_channels=config.unet_channels[i + 1],
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.BatchNorm2d(num_features=config.unet_channels[i + 1]),
                    nn.SiLU(inplace=True),
                )
            )
            self.up_second_convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=config.unet_channels[i + 1],
                        out_channels=config.unet_channels[i + 1],
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.BatchNorm2d(num_features=config.unet_channels[i + 1]),
                    nn.SiLU(inplace=True),
                )
            )
        self.out_cov = nn.Conv2d(
            in_channels=config.unet_channels[-1],
            out_channels=1,
            kernel_size=1,
        )

    def single_pass(
        self,
        condition: torch.Tensor,
        x1t: torch.Tensor,
        x2t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, ref2_dim, ref1_dim = condition.shape  # b, c, r2, r1
        x = torch.cat(
            (
                rearrange(
                    einsum(
                        F.one_hot(x1t.to(self.device), num_classes=ref1_dim),
                        F.one_hot(x2t.to(self.device), num_classes=ref2_dim),
                        "b r1, b r2 -> b r2 r1",
                    ),
                    "b r2 r1 -> b 1 r2 r1",
                ),
                condition.to(self.device),
            ),
            dim=1,
        )
        t_emb = self.time_emb(
            get_timestep_embedding(
                t.to(self.device),
                embedding_dim=self.config.unet_channels[0],
                flip_sin_to_cos=True,
                downscale_freq_shift=0,
            )
        )
        down_xs = []
        for i in range(len(self.down_first_convs)):
            down_xs.append(
                self.down_second_convs[i](
                    self.down_first_convs[i](x)
                    + rearrange(
                        self.down_time_embs[i](t_emb), "b c -> b c 1 1", b=batch_size
                    )
                )
            )
            x = self.down_samples[i](down_xs[-1])
        x = self.mid_second_conv(
            self.mid_first_conv(x)
            + rearrange(self.mid_time_emb(t_emb), "b c -> b c 1 1", b=batch_size)
        )
        for i in range(len(self.up_first_convs)):
            x = self.up_second_convs[i](
                self.up_first_convs[i](
                    torch.cat((down_xs.pop(), self.up_samples[i](x)), dim=1)
                )
                + rearrange(self.up_time_embs[i](t_emb), "b c -> b c 1 1", b=batch_size)
            )
        p_theta_0_on_t_logit = self.out_cov(x)
        return rearrange(p_theta_0_on_t_logit, "b 1 r2 r1 -> b r2 r1")

    def forward(self, input: dict, label: dict, my_generator: MyGenerator) -> dict:
        generator = my_generator.get_torch_generator_by_device(self.device)
        batch_size, _, ref2_dim, ref1_dim = input["condition"].shape  # b, c, r2, r1
        observation = torch.stack(
            [
                ob[
                    c2 - self.config.ext2_up : c2 + self.config.ext2_down + 1,
                    c1 - self.config.ext1_up : c1 + self.config.ext1_down + 1,
                ]
                for ob, c1, c2 in zip(
                    label["observation"], label["cut1"], label["cut2"]
                )
            ]
        ).to(self.device)
        t = torch.randint(
            1,
            self.config.noise_timesteps,
            (batch_size,),
            generator=generator,
            device=self.device,
        )
        # handle zero observation case
        x_cross0 = (
            Categorical(
                probs=rearrange(
                    observation + (observation.sum(dim=[1, 2], keepdim=True) == 0),
                    "b r2 r1 -> b (r2 r1)",
                )
            )
            .sample()
            .to(self.device)
        )
        x20 = x_cross0 // ref1_dim
        x10 = x_cross0 % ref1_dim
        x1t, x2t = self._add_noise(x10, x20, t, my_generator)

        p_theta_0_on_t_logit = self.single_pass(input["condition"], x1t, x2t, t)
        loss, loss_num = self.loss_fun(
            input["condition"],
            x10,
            x20,
            x1t,
            x2t,
            t,
            p_theta_0_on_t_logit,
            observation,
        )

        return {
            "p_theta_0_on_t_logit": p_theta_0_on_t_logit,
            "loss": loss,
            "loss_num": loss_num,
        }

    def loss_fun(
        self,
        condition: torch.Tensor,
        x10: torch.Tensor,
        x20: torch.Tensor,
        x1t: torch.Tensor,
        x2t: torch.Tensor,
        t: torch.Tensor,
        p_theta_0_on_t_logit: torch.Tensor,
        observation: torch.Tensor,
    ) -> float:
        loss = 0
        if "double_sample_negative_ELBO" in self.config.loss_weights:
            loss += (
                self._double_sample_negative_ELBO(condition, x1t, x2t, t)
                * self.config.loss_weights["double_sample_negative_ELBO"]
            )
        if "importance_sample_negative_ELBO" in self.config.loss_weights:
            loss += (
                self._importance_sample_negative_ELBO(
                    x10,
                    x20,
                    x1t,
                    x2t,
                    t,
                    p_theta_0_on_t_logit,
                )
            ) * self.config.loss_weights["importance_sample_negative_ELBO"]
        if "forward_negative_ELBO" in self.config.loss_weights:
            loss += (
                self._forward_negative_ELBO(x1t, x2t, t, p_theta_0_on_t_logit)
                * self.config.loss_weights["forward_negative_ELBO"]
            )
        if "reverse_negative_ELBO" in self.config.loss_weights:
            loss += (
                self._reverse_negative_ELBO(
                    x1t, x2t, t, p_theta_0_on_t_logit, observation
                )
                * self.config.loss_weights["reverse_negative_ELBO"]
            )
        if "sample_CE" in self.config.loss_weights:
            loss += (
                self._sample_CE(x10, x20, p_theta_0_on_t_logit)
                * self.config.loss_weights["sample_CE"]
            )
        if "non_sample_CE" in self.config.loss_weights:
            loss += (
                self._non_sample_CE(x1t, x2t, t, p_theta_0_on_t_logit, observation)
                * self.config.loss_weights["non_sample_CE"]
            )
        loss_num = observation.sum(dim=[1, 2]) * self._beta(t)
        loss = (loss * loss_num).sum()
        loss_num = loss_num.sum()
        return loss, loss_num

    def eval_output(self, examples: list[dict], batch: dict) -> pd.DataFrame:
        batch_size, _, ref2_dim, ref1_dim = batch["input"]["condition"].shape
        probas = []
        for i in tqdm(range(batch_size)):
            proba = torch.ones(ref2_dim, ref1_dim, device=self.device) / (
                ref2_dim * ref1_dim
            )
            for step in range(self.config.noise_timesteps - 1, 0, -1):
                p_theta_0_on_t_logit = self.single_pass(
                    condition=repeat(
                        batch["input"]["condition"][i],
                        "c r2 r1 -> b c r2 r1",
                        b=ref1_dim * ref2_dim,
                    ),
                    x1t=repeat(
                        torch.arange(ref1_dim),
                        "r1 -> (r2 r1)",
                        r2=ref2_dim,
                    ),
                    x2t=repeat(
                        torch.arange(ref2_dim),
                        "r2 -> (r2 r1)",
                        r1=ref1_dim,
                    ),
                    t=torch.full((ref1_dim * ref2_dim,), step),
                )
                p_theta_0_on_t = rearrange(
                    F.softmax(
                        rearrange(p_theta_0_on_t_logit, "b r2 r1 -> b (r2 r1)"),
                        dim=1,
                    ),
                    "b (r2 r1) -> b r2 r1",
                    r1=ref1_dim,
                    r2=ref2_dim,
                )
                q_tm1_on_0_t_1 = self._q_s_on_0_t(
                    t=torch.full((ref1_dim**2,), step, device=self.device),
                    s=torch.full((ref1_dim**2,), step - 1, device=self.device),
                    x0=repeat(
                        torch.arange(ref1_dim, device=self.device),
                        "r10 -> (r1t r10)",
                        r1t=ref1_dim,
                    ),
                    xt=repeat(
                        torch.arange(ref1_dim, device=self.device),
                        "r1t -> (r1t r10)",
                        r10=ref1_dim,
                    ),
                    stationary_sampler=self.stationary_sampler1,
                )
                q_tm1_on_0_t_2 = self._q_s_on_0_t(
                    t=torch.full((ref2_dim**2,), step, device=self.device),
                    s=torch.full((ref2_dim**2,), step - 1, device=self.device),
                    x0=repeat(
                        torch.arange(ref2_dim, device=self.device),
                        "r20 -> (r2t r20)",
                        r2t=ref2_dim,
                    ),
                    xt=repeat(
                        torch.arange(ref2_dim, device=self.device),
                        "r2t -> (r2t r20)",
                        r20=ref2_dim,
                    ),
                    stationary_sampler=self.stationary_sampler2,
                )
                proba = einsum(
                    proba,
                    rearrange(
                        p_theta_0_on_t,
                        "(r2t r1t) r20 r10 -> r2t r1t r20 r10",
                        r1t=ref1_dim,
                    ),
                    rearrange(
                        q_tm1_on_0_t_1, "(r1t r10) r1tm1 -> r1t r10 r1tm1", r1t=ref1_dim
                    ),
                    rearrange(
                        q_tm1_on_0_t_2, "(r2t r20) r2tm1 -> r2t r20 r2tm1", r2t=ref2_dim
                    ),
                    "r2t r1t, r2t r1t r20 r10, r1t r10 r1tm1, r2t r20 r2tm1 -> r2tm1 r1tm1",
                )
            probas.append(proba)
        probas = torch.stack(probas).cpu().numpy()

        df = pd.DataFrame(
            {
                "sample_idx": repeat(
                    np.arange(batch_size),
                    "b -> (b r2 r1)",
                    r1=ref1_dim,
                    r2=ref2_dim,
                ),
                "proba": probas.flatten(),
                "rpos1": repeat(
                    np.arange(-self.config.ext1_up, self.config.ext1_down + 1),
                    "r1 -> (b r2 r1)",
                    b=batch_size,
                    r2=ref2_dim,
                ),
                "rpos2": repeat(
                    np.arange(-self.config.ext2_up, self.config.ext2_down + 1),
                    "r2 -> (b r2 r1)",
                    b=batch_size,
                    r1=ref1_dim,
                ),
            }
        )
        return df

    def _q_s_on_0_t(
        self,
        t: torch.Tensor,
        s: torch.Tensor,
        x0: torch.Tensor,
        xt: torch.Tensor,
        stationary_sampler: Categorical,
    ):
        x0_one_hot = F.one_hot(x0, num_classes=stationary_sampler._num_events)
        xt_one_hot = F.one_hot(xt, num_classes=stationary_sampler._num_events)
        return (
            (
                einsum(self._alpha(t, s), xt_one_hot, "b, b r -> b r")
                + rearrange(
                    (1 - self._alpha(t, s))
                    * stationary_sampler.probs.to(self.device)[xt],
                    "b -> b 1",
                )
            )
            * (
                einsum(self._alpha(s), x0_one_hot, "b, b r -> b r")
                + einsum(
                    (1 - self._alpha(s)),
                    stationary_sampler.probs.to(self.device),
                    "b, r -> b r",
                )
            )
            / rearrange(
                self._alpha(t) * (xt == x0)
                + (1 - self._alpha(t)) * stationary_sampler.probs.to(self.device)[xt],
                "b -> b 1",
            )
        )

    def step(
        self,
        p_theta_0_on_t_logit: torch.Tensor,
        x1t: torch.Tensor,
        x2t: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple:
        x1t = x1t.to(self.device)
        x2t = x2t.to(self.device)
        t = t.to(self.device)
        s = torch.ceil(t) - 1
        batch_size, ref2_dim, ref1_dim = p_theta_0_on_t_logit.shape
        x_cross0 = Categorical(
            logits=p_theta_0_on_t_logit.view(batch_size, -1)
        ).sample()
        x20 = x_cross0 // ref1_dim
        x10 = x_cross0 % ref1_dim
        x1s = Categorical(
            probs=self._q_s_on_0_t(t, s, x10, x1t, self.stationary_sampler1)
        ).sample()
        x2s = Categorical(
            probs=self._q_s_on_0_t(t, s, x20, x2t, self.stationary_sampler2)
        ).sample()
        return x1s, x2s, s

    def _add_noise(
        self,
        x10: torch.Tensor,
        x20: torch.Tensor,
        t: torch.Tensor,
        my_generator: MyGenerator,
    ) -> tuple:
        generator = my_generator.get_torch_generator_by_device(self.device)
        # sample time and forward diffusion
        batch_size = t.shape[0]
        mask = torch.rand(
            batch_size, generator=generator, device=self.device
        ) < self._alpha(t)
        x1t = (
            x10 * mask
            + self.stationary_sampler1.sample((batch_size,)).to(self.device) * ~mask
        )
        mask = torch.rand(
            batch_size, generator=generator, device=self.device
        ) < self._alpha(t)
        x2t = (
            x20 * mask
            + self.stationary_sampler2.sample((batch_size,)).to(self.device) * ~mask
        )
        return x1t, x2t

    def _alpha(self, t: torch.Tensor, s: Optional[torch.Tensor] = None) -> torch.Tensor:
        if s is None:
            s = torch.zeros(t.shape, device=self.device)
        if self.config.noise_scheduler == "linear":
            return (self.config.noise_timesteps - t) / (
                self.config.noise_timesteps - s
            ).maximum(torch.tensor(torch.finfo(torch.float32).tiny))
        if self.config.noise_scheduler == "cosine":

            def cosine_frac(t: torch.Tensor) -> torch.Tensor:
                return torch.cos(
                    (t / self.config.noise_timesteps + self.config.cosine_factor)
                    / (1 + self.config.cosine_factor)
                    * torch.pi
                    / 2
                )

            return cosine_frac(t) / cosine_frac(s).maximum(
                torch.tensor(torch.finfo(torch.float32).tiny)
            )
        if self.config.noise_scheduler == "exp":
            return torch.exp(
                self.config.noise_timesteps
                * self.config.exp_scale
                * (
                    self.config.exp_base ** (s / self.config.noise_timesteps)
                    - self.config.exp_base ** (t / self.config.noise_timesteps)
                )
            )
        assert (
            self.config.noise_scheduler == "uniform"
        ), "supported noise schedulers are linear, cosine, exp, uniform"
        return torch.exp(self.config.uniform_scale * (s - t))

    def _beta(self, t: torch.Tensor) -> torch.Tensor:
        if self.config.noise_scheduler == "linear":
            return 1 / (self.config.noise_timesteps - t).maximum(
                torch.tensor(torch.finfo(torch.float32).tiny)
            )
        if self.config.noise_scheduler == "cosine":
            return (
                torch.pi
                * torch.tan(
                    (t / self.config.noise_timesteps + self.config.cosine_factor)
                    / (1 + self.config.cosine_factor)
                    * torch.pi
                    / 2
                )
                / (2 * self.config.noise_timesteps * (1 + self.config.cosine_factor))
            )
        if self.config.noise_scheduler == "exp":
            return (
                self.config.exp_scale
                * self.config.exp_base ** (t / self.config.noise_timesteps)
                * torch.log(torch.tensor(self.config.exp_base))
            )
        assert (
            self.config.noise_scheduler == "uniform"
        ), "supported noise schedulers are linear, cosine, exp, uniform"
        return torch.full(t.shape, self.config.uniform_scale, device=self.device)

    def _q_rkm_d(
        self,
        t: torch.Tensor,
        xt: torch.Tensor,
        stationary_sampler: Categorical,
    ) -> torch.Tensor:
        # xt: batch_size X ref_dim
        return einsum(
            self._alpha(t),
            F.one_hot(xt, stationary_sampler._num_events),
            "b, b r -> b r",
        ) + rearrange(
            (1 - self._alpha(t)) * stationary_sampler.probs.to(self.device)[xt],
            "b -> b 1",
        )

    def q_0_on_t(
        self,
        x1t: torch.Tensor,
        x2t: torch.Tensor,
        t: torch.Tensor,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, ref2_dim, ref1_dim = observation.shape
        q_rkm_1 = self._q_rkm_d(t, x1t, self.stationary_sampler1)
        q_rkm_2 = self._q_rkm_d(t, x2t, self.stationary_sampler2)
        return rearrange(
            F.normalize(
                rearrange(
                    einsum(
                        observation,
                        q_rkm_1,
                        q_rkm_2,
                        "b r2 r1, b r1, b r2 -> b r2 r1",
                    ),
                    "b r2 r1 -> b (r2 r1)",
                ),
                p=1.0,
                dim=1,
            ),
            "b (r2 r1) -> b r2 r1",
            r1=ref1_dim,
        )

    def _g_theta_d(
        self,
        t: torch.Tensor,
        xt: torch.Tensor,
        p_theta_0: torch.Tensor,
        dim: int,
        stationary_sampler: Categorical,
    ) -> torch.Tensor:
        auxilary_term = (
            1 + (1 / self._alpha(t) - 1) * stationary_sampler.probs.to(self.device)[xt]
        )
        xt_one_hot = F.one_hot(xt, stationary_sampler._num_events)
        p_theta_d_0 = p_theta_0.sum(dim=dim)
        return (
            einsum(
                (
                    1
                    - p_theta_d_0[torch.arange(p_theta_d_0.shape[0]), xt]
                    / auxilary_term
                ),
                stationary_sampler.probs.to(self.device),
                "b, r -> b r",
            )
            + einsum(
                self._alpha(t) / (1 - self._alpha(t)),
                p_theta_d_0,
                "b, b r -> b r",
            )
        ) * (1 - xt_one_hot) / rearrange(
            stationary_sampler.probs.to(self.device)[xt], "b -> b 1"
        ) + xt_one_hot

    def _sample_CE(
        self,
        x10: torch.Tensor,
        x20: torch.Tensor,
        p_theta_0_on_t_logit: torch.Tensor,
    ) -> float:
        batch_size, ref2_dim, ref1_dim = p_theta_0_on_t_logit.shape
        log_p_theta_0_on_t = rearrange(
            F.log_softmax(
                rearrange(
                    p_theta_0_on_t_logit,
                    "b r2 r1 -> b (r2 r1)",
                ),
                dim=1,
            ),
            "b (r2 r1) -> b r2 r1",
            r1=ref1_dim,
        )
        return -log_p_theta_0_on_t[torch.arange(batch_size), x20, x10]

    def _non_sample_CE(
        self,
        x1t: torch.Tensor,
        x2t: torch.Tensor,
        t: torch.Tensor,
        p_theta_0_on_t_logit: torch.Tensor,
        observation: torch.Tensor,
    ) -> float:
        batch_size, ref2_dim, ref1_dim = p_theta_0_on_t_logit.shape
        log_p_theta_0_on_t = rearrange(
            F.log_softmax(
                rearrange(
                    p_theta_0_on_t_logit,
                    "b r2 r1 -> b (r2 r1)",
                ),
                dim=1,
            ),
            "b (r2 r1) -> b r2 r1",
            r1=ref1_dim,
        )
        q_0_on_t = self.q_0_on_t(x1t, x2t, t, observation)
        return -einsum(log_p_theta_0_on_t, q_0_on_t, "b r2 r1, b r2 r1 -> b")

    def _common_negative_ELBO(
        self,
        x1t: torch.Tensor,
        x2t: torch.Tensor,
        t: torch.Tensor,
        p_theta_0_on_t_logit: torch.Tensor,
    ) -> tuple:
        batch_size, ref2_dim, ref1_dim = p_theta_0_on_t_logit.shape
        p_theta_0 = rearrange(
            F.softmax(
                rearrange(
                    p_theta_0_on_t_logit,
                    "b r2 r1 -> b (r2 r1)",
                ),
                dim=1,
            ),
            "b (r2 r1) -> b r2 r1",
            r1=ref1_dim,
        )

        g_theta_1_t = self._g_theta_d(t, x1t, p_theta_0, 1, self.stationary_sampler1)
        g_theta_2_t = self._g_theta_d(t, x2t, p_theta_0, 2, self.stationary_sampler2)

        return (
            einsum(
                self.stationary_sampler1.probs.to(self.device)[x1t],
                g_theta_1_t,
                "b, b r1 -> b",
            )
            + einsum(
                self.stationary_sampler2.probs.to(self.device)[x2t],
                g_theta_2_t,
                "b, b r2 -> b",
            ),
            g_theta_1_t,
            g_theta_2_t,
        )

    def _forward_negative_ELBO(
        self,
        x1t: torch.Tensor,
        x2t: torch.Tensor,
        t: torch.Tensor,
        p_theta_0_on_t_logit: torch.Tensor,
    ):
        common_negative_ELBO, g_theta_1_t, g_theta_2_t = self._common_negative_ELBO(
            x1t, x2t, t, p_theta_0_on_t_logit
        )

        return (
            common_negative_ELBO
            + einsum(
                self.stationary_sampler1.probs.to(self.device),
                g_theta_1_t.log().clamp_min(-1000),
                "r1, b r1 -> b",
            )
            + einsum(
                self.stationary_sampler2.probs.to(self.device),
                g_theta_2_t.log().clamp_min(-1000),
                "r2, b r2 -> b",
            )
        )

    def _reverse_negative_ELBO(
        self,
        x1t: torch.Tensor,
        x2t: torch.Tensor,
        t: torch.Tensor,
        p_theta_0_on_t_logit: torch.Tensor,
        observation: torch.Tensor,
    ):
        common_negative_ELBO, g_theta_1_t, g_theta_2_t = self._common_negative_ELBO(
            x1t, x2t, t, p_theta_0_on_t_logit
        )

        q_0_on_t = self.q_0_on_t(x1t, x2t, t, observation)

        g_1_t = self._g_theta_d(t, x1t, q_0_on_t, 1, self.stationary_sampler1)
        g_2_t = self._g_theta_d(t, x2t, q_0_on_t, 2, self.stationary_sampler2)

        return (
            common_negative_ELBO
            - einsum(g_1_t, g_theta_1_t.log().clamp_min(-1000), "b r1, b r1 -> b")
            - einsum(g_2_t, g_theta_2_t.log().clamp_min(-1000), "b r2, b r2 -> b")
        )

    def _double_sample_negative_ELBO(
        self,
        condition: torch.Tensor,
        x1t: torch.Tensor,
        x2t: torch.Tensor,
        t: torch.Tensor,
    ) -> float:
        batch_size, _, ref2_dim, ref1_dim = condition.shape
        barR1 = repeat(
            self.stationary_sampler1.probs.to(self.device), "r1 -> b r1", b=batch_size
        ).clone()
        barR1[torch.arange(batch_size), x1t] = 0
        barR2 = repeat(
            self.stationary_sampler2.probs.to(self.device), "r2 -> b r2", b=batch_size
        ).clone()
        barR2[torch.arange(batch_size), x2t] = 0
        yt = Categorical(probs=torch.cat((barR1, barR2), dim=1)).sample()
        mask = yt < ref1_dim
        y1t = mask * yt + ~mask * x1t
        y2t = ~mask * (yt - ref1_dim) + mask * x2t
        p_theta_0_on_t_logit = self.single_pass(condition, y1t, y2t, t)
        common_negative_ELBO, g_theta_1_t, g_theta_2_t = self._common_negative_ELBO(
            y1t, y2t, t, p_theta_0_on_t_logit
        )
        return common_negative_ELBO - (
            2
            - self.stationary_sampler1.probs.to(self.device)[x1t]
            - self.stationary_sampler2.probs.to(self.device)[x2t]
        ) * (
            self.stationary_sampler1.probs.to(self.device)[y1t]
            * (y1t != x1t)
            * g_theta_1_t[torch.arange(batch_size), x1t]
            + self.stationary_sampler2.probs.to(self.device)[y2t]
            * (y2t != x2t)
            * g_theta_2_t[torch.arange(batch_size), x2t]
        ).log().clamp_min(
            -1000
        )

    def _importance_sample_negative_ELBO(
        self,
        x10: torch.Tensor,
        x20: torch.Tensor,
        x1t: torch.Tensor,
        x2t: torch.Tensor,
        t: torch.Tensor,
        p_theta_0_on_t_logit: torch.Tensor,
    ) -> float:
        batch_size, ref2_dim, ref1_dim = p_theta_0_on_t_logit.shape
        common_negative_ELBO, g_theta_1_t, g_theta_2_t = self._common_negative_ELBO(
            x1t, x2t, t, p_theta_0_on_t_logit
        )

        q_t_on_0_1 = einsum(
            self.stationary_sampler1.probs.to(self.device),
            1 - self._alpha(t),
            "r1, b -> b r1",
        )
        q_t_on_0_1[torch.arange(batch_size), x10] += self._alpha(t)
        q_t_on_0_2 = einsum(
            self.stationary_sampler2.probs.to(self.device),
            1 - self._alpha(t),
            "r1, b -> b r1",
        )
        q_t_on_0_2[torch.arange(batch_size), x20] += self._alpha(t)

        return (
            common_negative_ELBO
            - einsum(q_t_on_0_1, g_theta_1_t.log().clamp_min(-1000), "b r1, b r1 -> b")
            / q_t_on_0_1[torch.arange(batch_size), x1t]
            * self.stationary_sampler1.probs.to(self.device)[x1t]
            - einsum(q_t_on_0_2, g_theta_2_t.log().clamp_min(-1000), "b r2, b r2 -> b")
            / q_t_on_0_2[torch.arange(batch_size), x2t]
            * self.stationary_sampler2.probs.to(self.device)[x2t]
        )

    @torch.no_grad()
    def reverse_diffusion(
        self,
        condition: torch.Tensor,
        sample_num: int,
        perfect_ob: Optional[torch.Tensor],
    ) -> list[tuple[np.ndarray]]:
        condition = repeat(
            condition,
            "c r2 r1 -> b c r2 r1",
            b=sample_num,
        )
        if perfect_ob is not None:
            perfect_ob = repeat(
                perfect_ob,
                "r2 r1 -> b r2 r1",
                b=sample_num,
            )
        x1t = self.stationary_sampler1.sample(sample_num)
        x2t = self.stationary_sampler2.sample(sample_num)
        t = torch.full((sample_num,), self.config.noise_timesteps - 1)
        path = [(x1t.cpu().numpy(), x2t.cpu().numpy())]
        for step in range(self.config.noise_timesteps - 1, 0, -1):
            if perfect_ob is None:
                p_theta_0_on_t_logit = self.single_pass(
                    condition,
                    x1t,
                    x2t,
                    t,
                )
            else:
                q_0_on_t = self.q_0_on_t(
                    x1t,
                    x2t,
                    t,
                    perfect_ob,
                )
                p_theta_0_on_t_logit = q_0_on_t.log().clamp_min(-1000)
            x1t, x2t, t = self.step(
                p_theta_0_on_t_logit,
                x1t,
                x2t,
                t,
            )
            path = [(x1t.cpu().numpy(), x2t.cpu().numpy())] + path

    @torch.no_grad()
    def draw_reverse_diffusion(
        self,
        path: list[tuple[np.ndarray]],
        filestem: str,
        interval: float = 120,
        pad: float = 5,
    ) -> None:
        fig, ax = plt.subplots()
        x1t, x2t = path[-1]
        scat = ax.scatter(
            x1t - self.config.ext1_up,
            x2t - self.config.ext2_up,
            c="b",
            s=5,
        )
        ax.set(
            xlim=[-self.config.ext1_up, self.config.ext1_down],
            ylim=[-self.config.ext2_up, self.config.ext2_down],
            xlabel="ref1",
            ylabel="ref2",
        )

        def update(frame):
            idx = min(len(path) + pad - frame, len(path) - 1)
            scat.set_offsets(np.stack(path[idx], axis=1))
            return scat

        ani = animation.FuncAnimation(
            fig=fig, func=update, frames=len(path) + pad, interval=interval
        )
        ani.save(filename=f"{filestem}.gif", writer="pillow")
        fig.savefig(f"{filestem}.png")
        plt.close()
