from .config import args, output_dir
from .model import model
from .optimizer import optimizer
from .learn_schedular import learn_scheduler
from .load_data import train_dataloader
from accelerate import Accelerator
import os

# Initialize accelerator and tensorboard logging
accelerator = Accelerator(
    mixed_precision=args.mixed_precision,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    log_with="tensorboard",
    project_dir=os.path.join(output_dir, "logs")
)

if accelerator.is_main_process:
    accelerator.init_trackers(args.data_file.as_posix())

# Prepare everything
# There is no specific order to remember, you just need to unpack the
# objects in the same order you gave them to the prepare method.
model, optimizer, train_dataloader, learn_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, learn_scheduler
)