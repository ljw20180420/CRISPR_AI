import torch.optim
from .config import args
from .diffuser_model import diffuser_model

if args.optimizer == "AdamW":
    optimizer = torch.optim.AdamW(
        diffuser_model.parameters(),
        lr=args.learning_rate,
        betas=args.adam_betas,
        eps=args.adam_eps,
        weight_decay=args.adam_weight_decay,
        amsgrad=args.adam_amsgrad,
    )