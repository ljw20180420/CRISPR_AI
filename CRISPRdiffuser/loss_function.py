import torch
from .model import stationary_sampler1, stationary_sampler2
import torch.nn.functional as F
from .config import args, ref1len, ref2len

def continuous_time_loss_function(alpha_t, x1t, x2t, p_theta_0, batch):
    def get_g_theta_d_t(stationary_sampler, xt, dim, reflen):
        auxilary_term = 1 + (1 / alpha_t - 1) * stationary_sampler.probs[xt]
        p_theta_d_0 = p_theta_0.sum(dim=dim)
        xt_one_hot = F.one_hot(xt, reflen + 1)
        return (
            (1 - p_theta_d_0[torch.arange(p_theta_d_0.shape[0]), xt] / auxilary_term).unsqueeze(1) * stationary_sampler.probs +
            (alpha_t / (1 - alpha_t)).unsqueeze(1) * p_theta_d_0
        ) * (1 - xt_one_hot) / stationary_sampler.probs[xt].unsqueeze(1) + xt_one_hot

    batch_size = p_theta_0.shape[0]
    g_theta_1_t = get_g_theta_d_t(stationary_sampler1, x1t, 1, ref1len)
    g_theta_2_t = get_g_theta_d_t(stationary_sampler2, x2t, 2, ref2len)

    return torch.inner(
        stationary_sampler1.probs[x1t] * g_theta_1_t.sum(dim = 1) +
        torch.inner(stationary_sampler1.probs.unsqueeze(0), torch.log(g_theta_1_t)).squeeze() +
        stationary_sampler2.probs[x2t] * g_theta_2_t.sum(dim = 1) +
        torch.inner(stationary_sampler2.probs.unsqueeze(0), torch.log(g_theta_2_t)).squeeze() -
        args.MCMC_corrector_factor * torch.log(p_theta_0[torch.arange(batch_size), batch['ref2_start'], batch['ref1_end']]),
        batch['weight']
    )
    

