from diffusers.models.embeddings import get_timestep_embedding
from transformers import PretrainedConfig, PreTrainedModel
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Optional, Literal

# torch does not import opt_einsum as backend by default. import opt_einsum manually will enable it.
from torch.backends import opt_einsum
from einops import einsum, rearrange, repeat


class CRIfuserConfig(PretrainedConfig):
    model_type = "CRIfuser"
    label_names = ["observation"]

    def __init__(
        self,
        ext1_up: Optional[int] = None,
        ext1_down: Optional[int] = None,
        ext2_up: Optional[int] = None,
        ext2_down: Optional[int] = None,
        max_micro_homology: Optional[int] = None,
        loss_weights: Optional[
            dict[
                Literal[
                    "double_sample_negative_ELBO",
                    "importance_sample_negative_ELBO",
                    "forward_negative_ELBO",
                    "reverse_negative_ELBO",
                    "sample_CE",
                    "non_sample_CE",
                ],
                float,
            ]
        ] = None,
        unet_channels: Optional[list[int]] = None,
        noise_scheduler: Optional[Literal["linear", "cosine", "exp", "uniform"]] = None,
        noise_timesteps: Optional[int] = None,
        cosine_factor: Optional[float] = None,
        exp_scale: Optional[float] = None,
        exp_base: Optional[float] = None,
        uniform_scale: Optional[float] = None,
        seed: Optional[int] = None,
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
        self.seed = seed
        super().__init__(**kwargs)


class CRIfuserModel(PreTrainedModel):
    config_class = CRIfuserConfig

    def __init__(self, config: CRIfuserConfig) -> None:
        super().__init__(config)
        # In more recent versions of PyTorch, you no longer need to explicitly register_parameter, it's enough to set a member of your nn.Module with nn.Parameter to "notify" pytorch that this variable should be treated as a trainable parameter (https://stackoverflow.com/questions/59234238/how-to-add-parameters-in-module-class-in-pytorch-custom-model).
        self.generator = torch.Generator().manual_seed(config.seed)
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
        self.initialize_weights()

    def initialize_weights(self) -> None:
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
                einsum(
                    x1t,
                    x2t,
                    "b r1, b r2 -> b 1 r2 r1",
                    b=batch_size,
                    r1=ref1_dim,
                    r2=ref2_dim,
                ),
                condition,
            ),
            dim=1,
        )
        t_emb = self.time_emb(
            get_timestep_embedding(
                t,
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
        return p_theta_0_on_t_logit

    def forward(
        self,
        condition: torch.Tensor,
        observation: Optional[torch.Tensor] = None,
    ):
        batch_size, _, ref2_dim, ref1_dim = condition.shape  # b, c, r2, r1
        t = torch.randint(
            1, self.config.noise_timesteps + 1, (batch_size,), generator=self.generator
        )
        if observation:
            x_cross0 = Categorical(probs=observation.view(batch_size, -1)).sample()
            x20 = x_cross0 // ref1_dim
            x10 = x_cross0 % ref1_dim
            x1t, x2t = self.add_noise(x10, x20, t)
        else:
            x1t = self.stationary_sampler1.sample(sample_shape=(batch_size,))
            x2t = self.stationary_sampler2.sample(sample_shape=(batch_size,))

        p_theta_0_on_t_logit = self.single_pass(condition, x1t, x2t, t)
        if observation:
            loss = 0
            if "double_sample_negative_ELBO" in self.config.loss_weights:
                loss += (
                    self.double_sample_negative_ELBO(condition, x1t, x2t, t)
                    * self.config.loss_weights["double_sample_negative_ELBO"]
                )
            if "importance_sample_negative_ELBO" in self.config.loss_weights:
                loss += (
                    self.importance_sample_negative_ELBO(
                        condition,
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
                    self.forward_negative_ELBO(x1t, x2t, t, p_theta_0_on_t_logit)
                    * self.config.loss_weights["forward_negative_ELBO"]
                )
            if "reverse_negative_ELBO" in self.config.loss_weights:
                loss += (
                    self.reverse_negative_ELBO(
                        x1t, x2t, t, p_theta_0_on_t_logit, observation
                    )
                    * self.config.loss_weights["reverse_negative_ELBO"]
                )
            if "sample_CE" in self.config.loss_weights:
                loss += (
                    self.sample_CE(x10, x20, p_theta_0_on_t_logit)
                    * self.config.loss_weights["sample_CE"]
                )
            if "non_sample_CE" in self.config.loss_weights:
                loss += (
                    self.non_sample_CE(x1t, x2t, t, p_theta_0_on_t_logit, observation)
                    * self.config.loss_weights["non_sample_CE"]
                )
            loss = (loss * observation.sum(dim=[1, 2]) * self.beta(t)).sum()
            return {
                "p_theta_0_on_t_logit": p_theta_0_on_t_logit,
                "loss": loss,
            }
        return {
            "p_theta_0_on_t_logit": p_theta_0_on_t_logit,
        }

    def q_s_on_0_t(
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
                einsum(self.alpha(t, s), xt_one_hot, "b, b r -> b r")
                + rearrange(
                    (1 - self.alpha(t, s)) * stationary_sampler.probs[xt],
                    "b -> b 1",
                )
            )
            * (
                einsum(self.alpha(s), x0_one_hot, "b, b r -> b r")
                + einsum((1 - self.alpha(s)), stationary_sampler.probs, "b, r -> b r")
            )
            / rearrange(
                self.alpha(t) * (xt == x0)
                + (1 - self.alpha(t)) * stationary_sampler.probs[xt],
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

        s = torch.ceil(t) - 1
        batch_size = p_theta_0_on_t_logit.shape[0]
        x_cross0 = Categorical(
            logits=p_theta_0_on_t_logit.view(batch_size, -1)
        ).sample()
        x20 = x_cross0 // (self.stationary_sampler1._num_events)
        x10 = x_cross0 % (self.stationary_sampler1._num_events)
        x1s = Categorical(
            probs=self.q_s_on_0_t(t, s, x10, x1t, self.stationary_sampler1)
        ).sample()
        x2s = Categorical(
            probs=self.q_s_on_0_t(t, s, x20, x2t, self.stationary_sampler2)
        ).sample()
        return x1s, x2s, s

    def add_noise(
        self,
        x10: torch.Tensor,
        x20: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple:
        # sample time and forward diffusion
        batch_size = t.shape[0]
        mask = torch.rand(batch_size) < self.alpha(t)
        x1t = (
            x10 * mask
            + self.stationary_sampler1.sample(torch.Size([batch_size])) * ~mask
        )
        mask = torch.rand(batch_size) < self.alpha(t)
        x2t = (
            x20 * mask
            + self.stationary_sampler2.sample(torch.Size([batch_size])) * ~mask
        )
        return x1t, x2t

    def alpha(self, t: torch.Tensor, s: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not s:
            s = torch.zeros(t.shape)
        if self.noise_scheduler == "linear":
            return (self.config.num_train_timesteps - t) / (
                self.config.num_train_timesteps - s
            ).maximum(torch.tensor(torch.finfo(torch.float32).tiny))
        if self.noise_scheduler == "cosine":

            def cosine_frac(t: torch.Tensor) -> torch.Tensor:
                return torch.cos(
                    (t / self.config.num_train_timesteps + self.config.cosine_factor)
                    / (1 + self.config.cosine_factor)
                    * torch.pi
                    / 2
                )

            return cosine_frac(t) / cosine_frac(s).maximum(
                torch.tensor(torch.finfo(torch.float32).tiny)
            )
        if self.noise_scheduler == "exp":
            return torch.exp(
                self.config.num_train_timesteps
                * self.config.exp_scale
                * (
                    self.config.exp_base ** (s / self.config.num_train_timesteps)
                    - self.config.exp_base ** (t / self.config.num_train_timesteps)
                )
            )
        assert (
            self.noise_scheduler == "uniform"
        ), "supported noise schedulers are linear, cosine, exp, uniform"
        return torch.exp(self.config.uniform_scale * (s - t))

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        if self.noise_scheduler == "linear":
            return 1 / (self.config.num_train_timesteps - t).maximum(
                torch.tensor(torch.finfo(torch.float32).tiny)
            )
        if self.noise_scheduler == "cosine":
            return (
                torch.pi
                * torch.tan(
                    (t / self.config.num_train_timesteps + self.config.cosine_factor)
                    / (1 + self.config.cosine_factor)
                    * torch.pi
                    / 2
                )
                / (
                    2
                    * self.config.num_train_timesteps
                    * (1 + self.config.cosine_factor)
                )
            )
        if self.noise_scheduler == "exp":
            return (
                self.config.exp_scale
                * self.config.exp_base ** (t / self.config.num_train_timesteps)
                * torch.log(self.config.exp_base)
            )
        assert (
            self.noise_scheduler == "uniform"
        ), "supported noise schedulers are linear, cosine, exp, uniform"
        return torch.full(t.shape, self.config.uniform_scale)

    def q_rkm_d(
        self,
        t: torch.Tensor,
        xt: torch.Tensor,
        stationary_sampler: Categorical,
    ) -> torch.Tensor:
        # xt: batch_size X ref_dim
        return einsum(
            self.alpha(t),
            F.one_hot(xt, stationary_sampler._num_events),
            "b, b r -> b r",
        ) + rearrange(
            (1 - self.alpha(t)) * stationary_sampler.probs[xt],
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
        q_rkm_1 = self.q_rkm_d(t, x1t, self.stationary_sampler1)
        q_rkm_2 = self.q_rkm_d(t, x2t, self.stationary_sampler2)
        return rearrange(
            F.normalize(
                einsum(
                    observation,
                    q_rkm_1,
                    q_rkm_2,
                    "b r2 r1, b r1, b r2 -> b (r2 r1)",
                    b=batch_size,
                    r1=ref1_dim,
                    r2=ref2_dim,
                ),
                p=1.0,
                dim=1,
            ),
            "b (r2 r1) -> b r2 r1",
            b=batch_size,
            r1=ref1_dim,
            r2=ref2_dim,
        )

    def g_theta_d(
        self,
        t: torch.Tensor,
        xt: torch.Tensor,
        p_theta_0: torch.Tensor,
        dim: int,
        stationary_sampler: Categorical,
    ) -> torch.Tensor:
        auxilary_term = 1 + (1 / self.alpha(t) - 1) * stationary_sampler.probs[xt]
        xt_one_hot = F.one_hot(xt, stationary_sampler._num_events)
        p_theta_d_0 = p_theta_0.sum(dim=dim)
        return (
            einsum(
                (
                    1
                    - p_theta_d_0[torch.arange(p_theta_d_0.shape[0]), xt]
                    / auxilary_term
                ),
                stationary_sampler.probs,
                "b, r -> b r",
            )
            + einsum(
                self.alpha(t) / (1 - self.alpha(t)),
                p_theta_d_0,
                "b, b r -> b r",
            )
        ) * (1 - xt_one_hot) / rearrange(
            stationary_sampler.probs[xt], "b -> b 1"
        ) + xt_one_hot

    def sample_CE(
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

    def non_sample_CE(
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

    def common_negative_ELBO(
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

        g_theta_1_t = self.g_theta_d(t, x1t, p_theta_0, 1, self.stationary_sampler1)
        g_theta_2_t = self.g_theta_d(t, x2t, p_theta_0, 2, self.stationary_sampler2)

        return (
            einsum(self.stationary_sampler1.probs[x1t], g_theta_1_t, "b, b r1 -> b")
            + einsum(self.stationary_sampler2.probs[x2t], g_theta_2_t, "b, b r2 -> b"),
            g_theta_1_t,
            g_theta_2_t,
        )

    def forward_negative_ELBO(
        self,
        x1t: torch.Tensor,
        x2t: torch.Tensor,
        t: torch.Tensor,
        p_theta_0_on_t_logit: torch.Tensor,
    ):
        common_negative_ELBO, g_theta_1_t, g_theta_2_t = self.common_negative_ELBO(
            x1t, x2t, t, p_theta_0_on_t_logit
        )

        return (
            common_negative_ELBO
            + einsum(
                self.stationary_sampler1.probs,
                g_theta_1_t.log().clamp_min(-1000),
                "r1, b r1 -> b",
            )
            + einsum(
                self.stationary_sampler2.probs,
                g_theta_2_t.log().clamp_min(-1000),
                "r2, b r2 -> b",
            )
        )

    def reverse_negative_ELBO(
        self,
        x1t: torch.Tensor,
        x2t: torch.Tensor,
        t: torch.Tensor,
        p_theta_0_on_t_logit: torch.Tensor,
        observation: torch.Tensor,
    ):
        common_negative_ELBO, g_theta_1_t, g_theta_2_t = self.common_negative_ELBO(
            x1t, x2t, t, p_theta_0_on_t_logit
        )

        q_0_on_t = q_0_on_t = self.q_0_on_t(x1t, x2t, t, observation)

        g_1_t = self.g_theta_d(t, x1t, q_0_on_t, 1, self.stationary_sampler1)
        g_2_t = self.g_theta_d(t, x2t, q_0_on_t, 2, self.stationary_sampler2)

        return (
            common_negative_ELBO
            - einsum(g_1_t, g_theta_1_t.log().clamp_min(-1000), "b r1, b r1 -> b")
            - einsum(g_2_t, g_theta_2_t.log().clamp_min(-1000), "b r2, b r2 -> b")
        )

    def double_sample_negative_ELBO(
        self,
        condition: torch.Tensor,
        x1t: torch.Tensor,
        x2t: torch.Tensor,
        t: torch.Tensor,
    ) -> float:
        batch_size, ref2_dim, ref1_dim = p_theta_0_on_t_logit.shape
        barR1 = repeat(self.stationary_sampler1.probs, "r1 -> b r1", b=batch_size)
        barR1[torch.arange(batch_size), x1t] = 0
        barR2 = repeat(self.stationary_sampler2.probs, "r2 -> b r2", b=batch_size)
        barR2[torch.arange(batch_size), x2t] = 0
        yt = Categorical(probs=torch.cat((barR1, barR2), dim=1)).sample()
        mask = yt < ref1_dim
        y1t = mask * yt + ~mask * x1t
        y2t = ~mask * (yt - ref1_dim) + mask * x2t
        p_theta_0_on_t_logit = self.single_pass(condition, x1t, x2t, t)
        common_negative_ELBO, g_theta_1_t, g_theta_2_t = self.common_negative_ELBO(
            y1t, y2t, t, p_theta_0_on_t_logit
        )
        return common_negative_ELBO - (
            2 - self.stationary_sampler1[x1t] - self.stationary_sampler2[x2t]
        ) * (
            self.stationary_sampler1[y1t]
            * (y1t != x1t)
            * g_theta_1_t[torch.arange(batch_size), x1t]
            + self.stationary_sampler2[y2t]
            * (y2t != x2t)
            * g_theta_2_t[torch.arange(batch_size), x2t]
        ).log().clamp_min(
            -1000
        )

    def importance_sample_negative_ELBO(
        self,
        condition: torch.Tensor,
        x10: torch.Tensor,
        x20: torch.Tensor,
        x1t: torch.Tensor,
        x2t: torch.Tensor,
        t: torch.Tensor,
        p_theta_0_on_t_logit: torch.Tensor,
    ) -> float:
        batch_size, ref2_dim, ref1_dim = p_theta_0_on_t_logit.shape
        common_negative_ELBO, g_theta_1_t, g_theta_2_t = self.common_negative_ELBO(
            x1t, x2t, t, p_theta_0_on_t_logit
        )

        q_t_on_0_1 = einsum(
            self.stationary_sampler1.probs, 1 - self.alpha(t), "r1, b -> b r1"
        )
        q_t_on_0_1[torch.arange(batch_size), x10] += self.alpha(t)
        q_t_on_0_2 = einsum(
            self.stationary_sampler2.probs, 1 - self.alpha(t), "r1, b -> b r1"
        )
        q_t_on_0_2[torch.arange(batch_size), x20] += self.alpha(t)

        return (
            common_negative_ELBO
            - einsum(q_t_on_0_1, g_theta_1_t.log().clamp_min(-1000), "b r1, b r1 -> b")
            / q_t_on_0_1[torch.arange(batch_size), x1t]
            * self.stationary_sampler1[x1t]
            - einsum(q_t_on_0_2, g_theta_2_t.log().clamp_min(-1000), "b r2, b r2 -> b")
            / q_t_on_0_2[torch.arange(batch_size), x2t]
            * self.stationary_sampler2[x2t]
        )
