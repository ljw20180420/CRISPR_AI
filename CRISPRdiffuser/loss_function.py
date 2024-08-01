import torch
from .model import stationary_sampler1, stationary_sampler2
import torch.nn.functional as F
from .config import args, ref1len, ref2len

def continuous_time_loss_function(alpha_t, x1t, x2t, p_theta_0, batch):
    def get_g_theta_d_and_q_rkm(stationary_sampler, xt, dim, reflen):
        auxilary_term = 1 + (1 / alpha_t - 1) * stationary_sampler.probs[xt]
        xt_one_hot = F.one_hot(xt, reflen + 1)
        p_theta_d_0 = p_theta_0.sum(dim=dim)
        g_theta_d = (
            (1 - p_theta_d_0[torch.arange(p_theta_d_0.shape[0]), xt] / auxilary_term)[:, None] * stationary_sampler.probs +
            (alpha_t / (1 - alpha_t))[:, None] * p_theta_d_0
        ) * (1 - xt_one_hot) / stationary_sampler.probs[xt][:, None] + xt_one_hot
        q_rkm = alpha_t[:, None] * xt_one_hot + ((1 - alpha_t) * stationary_sampler.probs[xt])[:, None]
        return g_theta_d, q_rkm

    batch_size = p_theta_0.shape[0]
    
    g_theta_1_t, q_rkm_1 = get_g_theta_d_and_q_rkm(stationary_sampler1, x1t, 1, ref1len)
    g_theta_2_t, q_rkm_2 = get_g_theta_d_and_q_rkm(stationary_sampler2, x2t, 2, ref2len)
    q_0_give_t = (batch['observation'] * q_rkm_1[:, None, :] * q_rkm_2[:, :, None]).view(batch_size, -1)
    q_0_give_t /= q_0_give_t.sum(dim=1, keepdim=True)

    return (
        stationary_sampler1.probs[x1t] * g_theta_1_t.sum(dim = 1) +
        torch.inner(stationary_sampler1.probs, g_theta_1_t.log()) +
        stationary_sampler2.probs[x2t] * g_theta_2_t.sum(dim = 1) +
        torch.inner(stationary_sampler2.probs, g_theta_2_t.log()) -
        args.MCMC_corrector_factor * (p_theta_0.log().view(batch_size, -1) * q_0_give_t).sum(dim=1)
    ).sum()
    

