import numpy as np
import pandas as pd
from diffusers import DiffusionPipeline
from transformers import PreTrainedModel
import torch
import torch.nn.functional as F
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# torch does not import opt_einsum as backend by default. import opt_einsum manually will enable it.
from torch.backends import opt_einsum
from einops import repeat, einsum, rearrange
from .load_data import DataCollator


@torch.no_grad()
class CRIfuserPipeline(DiffusionPipeline):
    # __init__ input name should be the same as the register module name
    def __init__(self, core_model: PreTrainedModel) -> None:
        super().__init__()

        self.register_modules(core_model=core_model)
        self.data_collator = DataCollator(
            ext1_up=core_model.config.ext1_up,
            ext1_down=core_model.config.ext1_down,
            ext2_up=core_model.config.ext2_up,
            ext2_down=core_model.config.ext2_down,
            max_micro_homology=core_model.config.max_micro_homology,
            output_label=True,
        )
        self.rpos2s = repeat(
            np.arange(
                -self.core_model.config.ext2_up,
                self.core_model.config.ext2_down + 1,
            ),
            "r2 -> r2 r1",
            r1=self.core_model.config.ext1_up + self.core_model.config.ext1_down + 1,
        ).flatten()

        self.rpos1s = repeat(
            np.arange(
                -self.core_model.config.ext1_up,
                self.core_model.config.ext1_down + 1,
            ),
            "r1 -> r2 r1",
            r2=self.core_model.config.ext2_up + self.core_model.config.ext2_down + 1,
        ).flatten()

    @torch.no_grad()
    def __call__(self, examples: list[dict], output_label: bool) -> dict:
        self.data_collator.output_label = output_label
        batch = self.data_collator(examples)
        batch_size = len(examples)
        ref1_dim = self.core_model.config.ext1_up + self.core_model.config.ext1_down + 1
        ref2_dim = self.core_model.config.ext2_up + self.core_model.config.ext2_down + 1
        probas = []
        for i in range(batch_size):
            proba = torch.ones(ref2_dim, ref1_dim) / (ref2_dim * ref1_dim)
            for step in range(self.core_model.config.noise_timesteps, 0, -1):
                p_theta_0_on_t_logit = self.core_model.single_pass(
                    condition=repeat(
                        batch["condition"][i],
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
                q_tm1_on_0_t_1 = self.core_model.q_s_on_0_t(
                    t=torch.full((ref1_dim**2,), step),
                    s=torch.full((ref1_dim**2,), step - 1),
                    x0=repeat(torch.arange(ref1_dim), "r10 -> (r1t r10)", r1t=ref1_dim),
                    xt=repeat(torch.arange(ref1_dim), "r1t -> (r1t r10)", r10=ref1_dim),
                    stationary_sampler=self.core_model.stationary_sampler1,
                )
                q_tm1_on_0_t_2 = self.core_model.q_s_on_0_t(
                    t=torch.full((ref2_dim**2,), step),
                    s=torch.full((ref2_dim**2,), step - 1),
                    x0=repeat(torch.arange(ref2_dim), "r20 -> (r2t r20)", r2t=ref2_dim),
                    xt=repeat(torch.arange(ref2_dim), "r2t -> (r2t r20)", r20=ref2_dim),
                    stationary_sampler=self.core_model.stationary_sampler2,
                )
                proba = einsum(
                    proba,
                    p_theta_0_on_t,
                    q_tm1_on_0_t_1,
                    q_tm1_on_0_t_2,
                    "r2t r1t, (r2t r1t) r20 r10, (r1t r10) r1tm1, (r2t r20) r2tm1 -> r2tm1 r1tm1",
                    r10=ref1_dim,
                    r20=ref2_dim,
                    r1t=ref1_dim,
                    r2t=ref2_dim,
                    r1tm1=ref1_dim,
                    r2tm1=ref2_dim,
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
                "rpos1": repeat(self.rpos1s, "(r2 r1) -> (b r2 r1)", b=batch_size),
                "rpos2": repeat(self.rpos2s, "(r2 r1) -> (b r2 r1)", b=batch_size),
            }
        )

        if output_label:
            loss = -(
                probas.log().clip(-1000, 0) * batch["observation"].cpu().numpy()
            ).sum()
            loss_num = batch["observation"].cpu().numpy().sum()
            return df, loss, loss_num
        return df

    @torch.no_grad()
    def inference(self, examples: list) -> dict:
        self.data_collator.output_label = False
        return self.__call__(
            examples=self.data_collator.inference(examples),
            output_label=False,
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
        x1t = self.core_model.stationary_sampler1.sample(sample_num)
        x2t = self.core_model.stationary_sampler2.sample(sample_num)
        t = torch.full((sample_num,), self.core_model.config.noise_timesteps)
        path = [(x1t.cpu().numpy(), x2t.cpu().numpy())]
        for step in range(self.core_model.config.noise_timesteps, 0, -1):
            if perfect_ob is None:
                p_theta_0_on_t_logit = self.core_model.single_pass(
                    condition,
                    x1t,
                    x2t,
                    t,
                )
            else:
                q_0_on_t = self.core_model.q_0_on_t(
                    x1t,
                    x2t,
                    t,
                    perfect_ob,
                )
                p_theta_0_on_t_logit = q_0_on_t.log().clamp_min(-1000)
            x1t, x2t, t = self.core_model.step(
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
        x1t, x2t = path[self.core_model.config.noise_timesteps]
        scat = ax.scatter(
            x1t - self.core_model.config.ext1_up,
            x2t - self.core_model.config.ext2_up,
            c="b",
            s=5,
        )
        ax.set(
            xlim=[-self.core_model.config.ext1_up, self.core_model.config.ext1_down],
            ylim=[-self.core_model.config.ext2_up, self.core_model.config.ext2_down],
            xlabel="ref1",
            ylabel="ref2",
        )

        def update(frame):
            idx = min(frame, len(path) - 1)
            scat.set_offsets(np.stack(path[idx], axis=1))
            return scat

        ani = animation.FuncAnimation(
            fig=fig, func=update, frames=len(path) + pad, interval=interval
        )
        ani.save(filename=f"{filestem}.gif", writer="pillow")
        fig.savefig(f"{filestem}.png")
        plt.close()
