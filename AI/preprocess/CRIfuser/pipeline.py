import numpy as np
from diffusers import DiffusionPipeline
from transformers import PreTrainedModel
import torch
from typing import Optional, Callable
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# torch does not import opt_einsum as backend by default. import opt_einsum manually will enable it.
from torch.backends import opt_einsum
from einops import repeat
from .load_data import DataCollator


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

    @torch.no_grad()
    def __call__(
        self, examples: list[dict], output_label: bool, metric: Optional[Callable]
    ) -> dict:
        self.data_collator.output_label = output_label
        batch = self.data_collator(examples)
        df = self.core_model.eval_output(batch)

        if output_label:
            assert metric is not None, "not metric given"
            loss, loss_num = metric(df=df, batch=batch)
            return df, loss, loss_num
        return df

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
        t = torch.full((sample_num,), self.core_model.config.noise_timesteps - 1)
        path = [(x1t.cpu().numpy(), x2t.cpu().numpy())]
        for step in range(self.core_model.config.noise_timesteps - 1, 0, -1):
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
        x1t, x2t = path[-1]
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
            idx = min(len(path) + pad - frame, len(path) - 1)
            scat.set_offsets(np.stack(path[idx], axis=1))
            return scat

        ani = animation.FuncAnimation(
            fig=fig, func=update, frames=len(path) + pad, interval=interval
        )
        ani.save(filename=f"{filestem}.gif", writer="pillow")
        fig.savefig(f"{filestem}.png")
        plt.close()
