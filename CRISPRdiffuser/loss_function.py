import torch

def negative_likelihood(logits, batch):
    return -torch.tensor([
        logits[i, batch['ref2_start'][i], batch['ref1_end'][i]] * batch['count'][i]
        for i in range(len(logits))
    ]).sum()
