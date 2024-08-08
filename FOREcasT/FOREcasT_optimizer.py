import tqdm
import torch
import torch.nn.functional as F

def computeKLObjAndGradients(theta, reg_coff, dataloader):
    # N is sgRNA number, M is mutation number, F is feature number
    # countss is NxM, datas is NxMxF, theta is F
    Q_reg = (reg_coff * (theta ** 2)).sum()
    grad_reg = reg_coff * theta
    Q, jac, total_size = 0, torch.zero(len(theta)), 0
    for batch in tqdm.tqdm(dataloader, desc="train batch loop"):
        logits = torch.log_softmax(torch.inner(batch['data'], theta), dim=1)
        fracs = batch['count'] / batch['count'].sum(dim=1, keepdims=True)
        Q += F.kl_div(logits, fracs, reduction='sum')
        jac += ((torch.exp(logits) - fracs)[:, None, :] @ batch['data']).squeeze().sum(dim=0)
        total_size += batch['count'].shape[0]
    return Q / total_size + Q_reg, jac / total_size + grad_reg