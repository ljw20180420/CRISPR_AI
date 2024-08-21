import torch.nn.functional as F

def kl_divergence(logits, counts, reg_coff, theta):
    return F.kl_div(
        logits,
        F.normalize(counts + 0.5, p=1.0, dim=1), # add 0.5 to prevent log(0), see loadOligoFeaturesAndReadCounts
        reduction='sum'
    ) + logits.shape[0] * (reg_coff * (theta ** 2)).sum()
    