from typing import List
from transformers import PretrainedConfig

class CRISPRDiffuserConfig(PretrainedConfig):
    model_type = "CRISPRdiffuser"
    label_names = ["observation"]
    main_input_name = "x1t_x2t_t"

    def __init__(
        self,
        channels: List = [13, 32, 64, 96, 64, 32, 1],
        MCMC_corrector_factor: float = 0.001,
        ref1len: int = 127,
        ref2len: int = 127,
        seed: int = 63036, # random seed for intialization
        **kwargs,
    ):
        self.channels = channels
        self.MCMC_corrector_factor = MCMC_corrector_factor
        self.ref1len = ref1len
        self.ref2len = ref2len
        self.seed = seed
        super().__init__(**kwargs)
