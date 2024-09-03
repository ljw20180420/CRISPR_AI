from transformers import PretrainedConfig

class CRISPRDiffuserConfig(PretrainedConfig):
    model_type = "CRISPRdiffuser"
    label_names = ["observation"]

    def __init__(
        self,
        channels = [13, 32, 64, 96, 64, 32, 1],
        MCMC_corrector_factor = 0.001,
        ref1len = 127,
        ref2len = 127,
        seed = 63036, # random seed for intialization
        **kwargs,
    ):
        self.channels = channels
        self.MCMC_corrector_factor = MCMC_corrector_factor
        self.ref1len = ref1len
        self.ref2len = ref2len
        self.seed = seed
        super().__init__(**kwargs)
