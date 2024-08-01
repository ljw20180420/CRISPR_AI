import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from .config import args, ref1len, ref2len, device
from .model import stationary_sampler1, stationary_sampler2
from .noise_scheduler import noise_scheduler
from datetime import datetime
from .save import save_heatmap

@torch.no_grad()
def inference_function(model, valid_step, batch, valid_epoch, epoch, ymd=f"{datetime.now():%Y-%m-%d}", save_dir=None):
    def get_q_s_0_t(x0, xt, xt_one_hot, stationary_sampler, reflen):
        return (
            alpha_ts[:, None] * xt_one_hot + ((1 - alpha_ts) * stationary_sampler.probs[xt])[:, None]
        ) * (
            alpha_s[:, None] * F.one_hot(x0, num_classes=reflen + 1) + (1 - alpha_s)[:, None] * stationary_sampler.probs
        ) / (
            alpha_t * (xt == x0) + (1 - alpha_t) * stationary_sampler.probs[xt]
        )[:, None]

    valid_batch_size = batch['condition'].shape[0]
    cumsum_p_theta_0 = torch.zeros(valid_batch_size, (ref2len + 1) * (ref1len + 1), device=device)
    x1t = stationary_sampler1.sample(torch.Size([valid_batch_size]))
    x2t = stationary_sampler2.sample(torch.Size([valid_batch_size]))
    for i in range(args.noise_timesteps, 0, -1):
        t = noise_scheduler(torch.tensor([i], device=device))
        s = noise_scheduler(torch.tensor([i - 1], device=device))
        alpha_ts = torch.e ** (s - t)
        alpha_s = torch.e ** (-s)
        alpha_t = torch.e ** (-t)
        x1t_one_hot = F.one_hot(x1t, num_classes=ref1len + 1)
        x2t_one_hot = F.one_hot(x2t, num_classes=ref2len + 1)
        values = x1t_one_hot.view(valid_batch_size, 1, -1) * x2t_one_hot.view(valid_batch_size, -1, 1)
        p_theta_0_logit = model(
            torch.cat((
                values[:, None, :, :],
                batch['condition']
            ), dim = 1),
            t
        )
        p_theta_0_sampler = Categorical(logits=p_theta_0_logit.view(valid_batch_size, -1))
        cumsum_p_theta_0 += p_theta_0_sampler.probs
        if save_dir:
            for in_batch in range(0, valid_batch_size, args.save_image_valid_in_batchs):
                save_heatmap(p_theta_0_sampler.probs[in_batch].view(ref2len + 1, -1), save_dir / ymd / f"epoch{epoch}" / f"valid_epoch{valid_epoch}" / f"valid_batch{valid_step}" / f"in_batch{in_batch}" / f"{i - 1}.png")
        x_cross0 = p_theta_0_sampler.sample()
        x20 = x_cross0 // (ref1len + 1)
        x10 = x_cross0 % (ref1len + 1)
        q_1_s_0_t = get_q_s_0_t(x10, x1t, x1t_one_hot, stationary_sampler1, ref1len)
        q_2_s_0_t = get_q_s_0_t(x20, x2t, x2t_one_hot, stationary_sampler2, ref2len)
        x1t = Categorical(probs=q_1_s_0_t).sample()
        x2t = Categorical(probs=q_2_s_0_t).sample()

    return (
        ((cumsum_p_theta_0 / args.noise_timesteps).log() * batch['observation'].view(valid_batch_size, -1)).sum(dim=1) /
        batch['observation'].view(valid_batch_size, -1).sum(dim=1)
    ).sum()
