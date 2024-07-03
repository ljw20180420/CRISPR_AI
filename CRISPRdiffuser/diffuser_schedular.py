import diffusers.schedulers
from .config import args

if args.diffuser_scheduler == "DPMSolverMultistepSchedule":
    # TODO: set more parameters for diffuser scheduler
    diffuser_scheduler = diffusers.schedulers.DPMSolverMultistepScheduler(
        num_train_timesteps=1000
    )
    