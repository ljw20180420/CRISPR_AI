from diffusers import DiffusionPipeline

class CRISPRDiffuserPipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler):
        super().__init__()

        self.register_modules(unet=unet, scheduler=scheduler)

    def __call__(self, condition, observation, batch_size=1):
        # sample from observation of batch_size
        # generate t
        # add noise
        # loop for denoise