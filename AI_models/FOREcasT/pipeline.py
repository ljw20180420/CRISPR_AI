import torch
from diffusers import DiffusionPipeline
import torch.nn.functional as F
import numpy as np

class FOREcasTPipeline(DiffusionPipeline):
    def __init__(self, FOREcasT_model, MAX_DEL_SIZE):
        super().__init__()

        self.register_modules(FOREcasT_model=FOREcasT_model)
        self.MAX_DEL_SIZE = MAX_DEL_SIZE
        self.lefts = np.concatenate([
            np.arange(-DEL_SIZE, 1)
            for DEL_SIZE in range(self.MAX_DEL_SIZE, -1, -1)
        ] + [np.zeros(20, np.int64)])
        self.rights = np.concatenate([
            np.arange(0, DEL_SIZE + 1)
            for DEL_SIZE in range(self.MAX_DEL_SIZE, -1, -1)
        ] + [np.zeros(20, np.int64)])
        self.inss = (self.MAX_DEL_SIZE + 2) * (self.MAX_DEL_SIZE + 1) // 2 * [""] + ["A", "C", "G", "T", "AA", "AC", "AG", "AT", "CA", "CC", "CG", "CT", "GA", "GC", "GG", "GT", "TA", "TC", "TG", "TT"]

    @torch.no_grad()
    def __call__(self, batch):
        assert batch["feature"].shape[1] == len(self.lefts), "the possible mutation number of the input feature does not fit the pipeline"
        return {
            "proba": F.softmax(self.FOREcasT_model(batch["feature"].to(self.FOREcasT_model.device))["logit"], dim=-1),
            "left": self.lefts,
            "right": self.rights,
            "ins_seq": self.inss
        }