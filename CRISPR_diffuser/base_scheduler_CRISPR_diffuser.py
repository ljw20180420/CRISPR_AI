from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
import torch
import torch.nn.functional as F
from typing import Union, Tuple
from torch.distributions import Categorical

class CRISPRDiffuserBaseScheduler(SchedulerMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self
    ):
        pass

    def step(
        self,
        model_output: torch.Tensor,
        x1t: torch.Tensor,
        x2t: torch.Tensor,
        t: torch.Tensor,
        stationary_sampler1: Categorical,
        stationary_sampler2: Categorical
    ) -> Tuple:
        def get_q_s_0_t(x0, xt, stationary_sampler):
            x0_one_hot = F.one_hot(x0, num_classes=stationary_sampler._num_events)
            xt_one_hot = F.one_hot(xt, num_classes=stationary_sampler._num_events)
            return (
                alpha_ts[:, None] * xt_one_hot + ((1 - alpha_ts) * stationary_sampler.probs[xt])[:, None]
            ) * (
                alpha_s[:, None] * x0_one_hot + (1 - alpha_s)[:, None] * stationary_sampler.probs
            ) / (
                alpha_t * (xt == x0) + (1 - alpha_t) * stationary_sampler.probs[xt]
            )[:, None]

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
        assert (timestep > 0).all(), "timestep must be positive to get previous timestep"
        # *** RuntimeError: "argmax_cpu" not implemented for 'Bool'
        index = (self.timesteps[None, :] < timestep[:, None]).to(torch.int8).argmax(dim=1)
        return self.timesteps[index]