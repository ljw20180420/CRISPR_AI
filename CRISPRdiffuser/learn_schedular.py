import diffusers.optimization
from .config import args
from .optimizer import optimizer
from .load_data import num_training_steps

learn_scheduler = diffusers.optimization.get_scheduler(
    name=args.learn_scheduler,
    optimizer=optimizer,
    step_rules=args.step_rules,
    num_warmup_steps=args.num_warmup_steps,
    num_training_steps=num_training_steps,
    num_cycles=args.num_cycles,
    power=args.power
)