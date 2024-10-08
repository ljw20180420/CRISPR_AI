from diffusers import DiffusionPipeline, __version__
from torch.distributions import Categorical
import torch
from tqdm import tqdm
import json
import numpy as np
from pathlib import Path

class CRISPRDiffuserPipeline(DiffusionPipeline):
    def to_json_string(self) -> str:
        """
        Serializes the configuration instance to a JSON string.

        Returns:
            `str`:
                String containing all the attributes that make up the configuration instance in JSON format.
        """
        config_dict = self._internal_dict if hasattr(self, "_internal_dict") else {}
        config_dict["_class_name"] = [self.__module__, self.__class__.__name__]
        config_dict["_diffusers_version"] = __version__

        def to_json_saveable(value):
            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, Path):
                value = value.as_posix()
            return value

        config_dict = {k: to_json_saveable(v) for k, v in config_dict.items()}
        # Don't save "_ignore_files" or "_use_default_values"
        config_dict.pop("_ignore_files", None)
        config_dict.pop("_use_default_values", None)

        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def __init__(self, unet, scheduler):
        super().__init__()

        self.register_modules(unet=unet, scheduler=scheduler)
        self.stationary_sampler1 = Categorical(probs=unet.stationary_sampler1_probs)
        self.stationary_sampler2 = Categorical(probs=unet.stationary_sampler2_probs)

    @torch.no_grad()
    def __call__(self, condition, batch_size=1, record_path=False):
        x1t = self.stationary_sampler1.sample(torch.Size([batch_size]))
        x2t = self.stationary_sampler2.sample(torch.Size([batch_size]))
        t = self.scheduler.step_to_time(torch.tensor([self.scheduler.config.num_train_timesteps]))
        if record_path:
            x1ts, x2ts, ts = [x1t], [x2t], [t]
        for timestep in tqdm(self.scheduler.timesteps):
            if timestep >= t:
                continue
            p_theta_0_logit = self.unet({"x1t": x1t, "x2t": x2t, "t": t}, condition.expand(batch_size, -1, -1, -1))["p_theta_0_logit"]
            # the scheduler automatically set t = timestep
            x1t, x2t, t = self.scheduler.step(p_theta_0_logit, x1t, x2t, t, self.stationary_sampler1, self.stationary_sampler2)
            if record_path:
                x1ts.append(x1t)
                x2ts.append(x2t)
                ts.append(t)
        if record_path:
            return x1ts, x2ts, ts
        return x1t, x2t
