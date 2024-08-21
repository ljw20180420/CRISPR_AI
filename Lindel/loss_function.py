import torch.nn.functional as F
from config import args

def cross_entropy_reg(logits, counts, theta):
    if args.reg_mode == "l2":
        reg_term = (theta ** 2).sum()
    elif args.reg_mode == "l1":
        reg_term = abs(theta).sum()
    return (F.log_softmax(logits, dim=1) * F.normalize(counts, p=1.0, dim=1)).sum() + logits.shape[0] * args.reg_const * reg_term
    