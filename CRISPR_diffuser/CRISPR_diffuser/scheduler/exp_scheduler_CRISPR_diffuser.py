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
        exp_base: float = 5.0,
        device: Union[str, torch.device] = None
    ):
        self.set_timesteps()

    def set_timesteps(
        self,
        num_inference_steps: int | None = None
    ):
        if num_inference_steps is None:
            num_inference_steps = self.config.num_train_timesteps
        t = torch.arange(num_inference_steps, -1, -1, device=self.config.device)
        self.timesteps = self.config.exp_scale * (self.config.exp_base ** (t / num_inference_steps) - 1)