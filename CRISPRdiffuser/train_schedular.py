import diffusers.schedulers
from .config import args

if args.diffuser_scheduler == "DDPMScheduler":
    # TODO: set more parameters for diffuser scheduler
    train_scheduler = diffusers.schedulers.DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps
    )
    