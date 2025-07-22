import numpy as np
import torch


class MyGenerator:
    def __init__(self, seed: int) -> None:
        self.seed = seed

    def __call__(self) -> None:
        self.np_rng = np.random.default_rng(self.seed)
        self.torch_c_rng = torch.Generator(device="cpu").manual_seed(self.seed)
        self.torch_g_rng = torch.Generator(device="cuda").manual_seed(self.seed)
