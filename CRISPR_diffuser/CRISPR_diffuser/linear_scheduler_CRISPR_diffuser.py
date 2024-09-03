from typing import Union
from diffusers.configuration_utils import register_to_config
import torch
from .base_scheduler_CRISPR_diffuser import CRISPRDiffuserBaseScheduler

class CRISPRDiffuserLinearScheduler(CRISPRDiffuserBaseScheduler):
    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 20,
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
        self.timesteps = (
            num_inference_steps / (num_inference_steps - t).maximum(torch.tensor(torch.finfo(torch.float32).tiny, device=self.config.device))
        ).log()