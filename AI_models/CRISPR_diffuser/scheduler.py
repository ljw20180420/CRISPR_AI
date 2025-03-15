from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
import torch
import torch.nn.functional as F
from typing import Tuple
from torch.distributions import Categorical


class CRISPRDiffuserBaseScheduler(SchedulerMixin, ConfigMixin):
    @register_to_config
    def __init__(self):
        pass

    def step(
        self,
        model_output: torch.Tensor,
        x1t: torch.Tensor,
        x2t: torch.Tensor,
        t: torch.Tensor,
        stationary_sampler1: Categorical,
        stationary_sampler2: Categorical,
    ) -> Tuple:
        def get_q_s_0_t(x0, xt, stationary_sampler):
            x0_one_hot = F.one_hot(x0, num_classes=stationary_sampler._num_events)
            xt_one_hot = F.one_hot(xt, num_classes=stationary_sampler._num_events)
            return (
                (
                    alpha_ts[:, None] * xt_one_hot
                    + ((1 - alpha_ts) * stationary_sampler.probs[xt])[:, None]
                )
                * (
                    alpha_s[:, None] * x0_one_hot
                    + (1 - alpha_s)[:, None] * stationary_sampler.probs
                )
                / (alpha_t * (xt == x0) + (1 - alpha_t) * stationary_sampler.probs[xt])[
                    :, None
                ]
            )

        s = self.previous_timestep(t)
        alpha_ts = torch.e ** (s - t)
        alpha_s = torch.e ** (-s)
        alpha_t = torch.e ** (-t)
        p_theta_0_logit = model_output
        batch_size = model_output.shape[0]
        x_cross0 = Categorical(logits=p_theta_0_logit.view(batch_size, -1)).sample()
        x20 = x_cross0 // (stationary_sampler1._num_events)
        x10 = x_cross0 % (stationary_sampler1._num_events)
        q_1_s_0_t = get_q_s_0_t(x10, x1t, stationary_sampler1)
        q_2_s_0_t = get_q_s_0_t(x20, x2t, stationary_sampler2)
        x1s = Categorical(probs=q_1_s_0_t).sample()
        x2s = Categorical(probs=q_2_s_0_t).sample()
        return x1s, x2s, s

    def add_noise(
        self,
        x10: torch.Tensor,
        x20: torch.Tensor,
        t: torch.Tensor,
        stationary_sampler1: Categorical,
        stationary_sampler2: Categorical,
    ) -> Tuple:
        # sample time and forward diffusion
        batch_size = t.shape[0]
        alpha_t = torch.e ** (-t)
        mask = torch.rand(batch_size) < alpha_t
        x1t = x10 * mask + stationary_sampler1.sample(torch.Size([batch_size])) * ~mask
        mask = torch.rand(batch_size) < alpha_t
        x2t = x20 * mask + stationary_sampler2.sample(torch.Size([batch_size])) * ~mask
        return x1t, x2t

    def __len__(self):
        return self.config.num_train_timesteps

    def previous_timestep(self, timestep: torch.Tensor):
        assert (
            timestep > 0
        ).all(), "timestep must be positive to get previous timestep"
        # *** RuntimeError: "argmax_cpu" not implemented for 'Bool'
        index = (
            (self.timesteps[None, :] < timestep[:, None]).to(torch.int8).argmax(dim=1)
        )
        return self.timesteps[index]


class CRISPRDiffuserCosineScheduler(CRISPRDiffuserBaseScheduler):
    @register_to_config
    def __init__(self, num_train_timesteps: int = 20, cosine_factor: float = 0.008):
        self.set_timesteps()

    def set_timesteps(self, num_inference_steps: int | None = None):
        if num_inference_steps is None:
            num_inference_steps = self.config.num_train_timesteps
        assert (
            num_inference_steps <= self.config.num_train_timesteps
        ), "inference steps exceed train steps"
        steps = torch.arange(num_inference_steps, -1, -1)
        self.timesteps = self.step_to_time(steps)

    def step_to_time(self, steps: torch.Tensor):
        return (
            torch.cos(
                torch.tensor(
                    self.config.cosine_factor
                    / (1 + self.config.cosine_factor)
                    * torch.pi
                    / 2
                )
            )
            / torch.cos(
                (steps / self.config.num_train_timesteps + self.config.cosine_factor)
                / (1 + self.config.cosine_factor)
                * torch.pi
                / 2
            ).maximum(torch.tensor(torch.finfo(torch.float32).tiny))
        ).log()


class CRISPRDiffuserExpScheduler(CRISPRDiffuserBaseScheduler):
    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 20,
        exp_scale: float = 5.0,
        exp_base: float = 5.0,
    ):
        self.set_timesteps()

    def set_timesteps(self, num_inference_steps: int | None = None):
        if num_inference_steps is None:
            num_inference_steps = self.config.num_train_timesteps
        assert (
            num_inference_steps <= self.config.num_train_timesteps
        ), "inference steps exceed train steps"
        steps = torch.arange(num_inference_steps, -1, -1)
        self.timesteps = self.step_to_time(steps)

    def step_to_time(self, steps: torch.Tensor):
        return self.config.exp_scale * (
            self.config.exp_base ** (steps / self.config.num_train_timesteps) - 1
        )


class CRISPRDiffuserLinearScheduler(CRISPRDiffuserBaseScheduler):
    @register_to_config
    def __init__(self, num_train_timesteps: int = 20):
        self.set_timesteps()

    def set_timesteps(self, num_inference_steps: int | None = None):
        if num_inference_steps is None:
            num_inference_steps = self.config.num_train_timesteps
        assert (
            num_inference_steps <= self.config.num_train_timesteps
        ), "inference steps exceed train steps"
        steps = torch.arange(num_inference_steps, -1, -1)
        self.timesteps = self.step_to_time(steps)

    def step_to_time(self, steps: torch.Tensor):
        (
            self.config.num_train_timesteps
            / (self.config.num_train_timesteps - steps).maximum(
                torch.tensor(torch.finfo(torch.float32).tiny)
            )
        ).log()


class CRISPRDiffuserUniformScheduler(CRISPRDiffuserBaseScheduler):
    @register_to_config
    def __init__(self, num_train_timesteps: int = 20, uniform_scale: float = 1.0):
        self.set_timesteps()

    def set_timesteps(self, num_inference_steps: int | None = None):
        if num_inference_steps is None:
            num_inference_steps = self.config.num_train_timesteps
        assert (
            num_inference_steps <= self.config.num_train_timesteps
        ), "inference steps exceed train steps"
        steps = torch.arange(num_inference_steps, -1, -1)
        self.timesteps = self.step_to_time(steps)

    def step_to_time(self, steps: torch.Tensor):
        return self.config.uniform_scale * steps / self.config.num_train_timesteps


def scheduler(
    noise_scheduler="exp",
    noise_timesteps=20,
    cosine_factor=0.008,
    exp_scale=5.0,
    exp_base=5.0,
    uniform_scale=1.0,
):
    if noise_scheduler == "linear":
        from .scheduler import CRISPRDiffuserLinearScheduler

        return CRISPRDiffuserLinearScheduler(num_train_timesteps=noise_timesteps)
    if noise_scheduler == "cosine":
        from .scheduler import CRISPRDiffuserCosineScheduler

        return CRISPRDiffuserCosineScheduler(
            num_train_timesteps=noise_timesteps, cosine_factor=cosine_factor
        )
    if noise_scheduler == "exp":
        from .scheduler import CRISPRDiffuserExpScheduler

        return CRISPRDiffuserExpScheduler(
            num_train_timesteps=noise_timesteps, exp_scale=exp_scale, exp_base=exp_base
        )
    if noise_scheduler == "uniform":
        from .scheduler import CRISPRDiffuserUniformScheduler

        return CRISPRDiffuserUniformScheduler(
            num_train_timesteps=noise_timesteps, uniform_scale=uniform_scale
        )
