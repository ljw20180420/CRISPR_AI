from diffusers import DiffusionPipeline
from torch.distributions import Categorical
import torch
from tqdm import tqdm

class CRISPRDiffuserPipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler):
        super().__init__()

        self.register_modules(unet=unet, scheduler=scheduler)
        self.stationary_sampler1 = Categorical(probs=unet.stationary_sampler1_probs)
        self.stationary_sampler2 = Categorical(probs=unet.stationary_sampler2_probs)

    @torch.no_grad()
    def __call__(self, batch, batch_size=1, record_path=False):
        x1t = self.stationary_sampler1.sample(torch.Size([batch_size]))
        x2t = self.stationary_sampler2.sample(torch.Size([batch_size]))
        t = self.scheduler.step_to_time(torch.tensor([self.scheduler.config.num_train_timesteps]))
        if record_path:
            x1ts, x2ts, ts = [x1t], [x2t], [t]
        for timestep in tqdm(self.scheduler.timesteps):
            if timestep >= t:
                continue
            p_theta_0_logit = self.unet(
                {
                    "x1t": x1t.to(self.unet.device),
                    "x2t": x2t.to(self.unet.device),
                    "t": t.to(self.unet.device)
                },
                batch["condition"].to(self.unet.device).expand(batch_size, -1, -1, -1)
            )["p_theta_0_logit"].cpu()
            # the scheduler automatically set t = timestep
            x1t, x2t, t = self.scheduler.step(p_theta_0_logit, x1t, x2t, t, self.stationary_sampler1, self.stationary_sampler2)
            if record_path:
                x1ts.append(x1t)
                x2ts.append(x2t)
                ts.append(t)
        if record_path:
            return x1ts, x2ts, ts
        return x1t, x2t
