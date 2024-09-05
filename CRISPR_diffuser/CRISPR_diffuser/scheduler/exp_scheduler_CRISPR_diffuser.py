from typing import Union
from diffusers.configuration_utils import register_to_config
import torch
from .base_scheduler_CRISPR_diffuser import CRISPRDiffuserBaseScheduler

class CRISPRDiffuserExpScheduler(CRISPRDiffuserBaseScheduler):
    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 20,
        exp_scale: float = 5.0,
        exp_base: float = 5.0
    ):
        self.set_timesteps()

    def set_timesteps(
        self,
        num_inference_steps: int | None = None
    ):
        if num_inference_steps is None:
            num_inference_steps = self.config.num_train_timesteps
        assert num_inference_steps <= self.config.num_train_timesteps, "inference steps exceed train steps"
        steps = torch.arange(num_inference_steps, -1, -1)
        self.timesteps = self.step_to_time(steps)

    def step_to_time(self, steps: torch.Tensor):
        return self.config.exp_scale * (self.config.exp_base ** (steps / self.config.num_train_timesteps) - 1)