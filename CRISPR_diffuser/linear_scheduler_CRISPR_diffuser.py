from typing import Union
from diffusers.configuration_utils import register_to_config
import torch
from base_scheduler_CRISPR_diffuser import CRISPRDiffuserBaseScheduler

class CRISPRDiffuserLinearScheduler(CRISPRDiffuserBaseScheduler):
    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 20
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
        (
            self.config.num_train_timesteps / (self.config.num_train_timesteps - steps).maximum(torch.tensor(torch.finfo(torch.float32).tiny))
        ).log()
