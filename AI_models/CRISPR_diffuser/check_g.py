import torch
import torch.nn.functional as F

ob_ref1 = [110, 103, 96, 103, 100, 98, 106, 105, 96, 101, 105, 109, 113, 90, 95, 99, 102, 111, 95, 96, 99, 100, 101, 113, 86, 92, 94, 95, 99, 100, 101, 102, 103, 107, 89, 92, 98, 100, 101, 114, 81, 85, 89, 91, 92, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 110, 91, 92, 93, 94, 95, 96, 97, 99, 100, 101, 102, 103, 104, 106, 110, 123, 87, 88, 89, 91, 93, 94, 95, 97, 98, 99, 100, 101, 103, 83, 91, 92, 93, 94, 95, 96, 99, 100, 102, 87, 88, 90, 91, 93, 95, 96, 97, 98, 99, 100, 101, 103, 105, 87, 90, 91, 98, 99, 100, 86, 96, 99, 100, 101, 103, 91, 94, 99, 100, 101, 103, 86, 90, 92, 94, 100, 101, 88, 90, 91, 92, 95, 98, 90, 93, 94, 95, 97, 99, 100, 102, 89, 91, 94, 95, 96, 97, 100, 97, 99, 100, 101, 97, 99, 93, 99, 100, 97, 100, 102, 105]
ob_ref2 = [2, 5, 13, 13, 15, 16, 18, 19, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 34, 34, 34, 34, 34, 34, 35, 35, 35, 35, 35, 35, 36, 36, 36, 36, 36, 36, 37, 37, 37, 37, 37, 37, 37, 37, 38, 38, 38, 38, 38, 38, 38, 39, 39, 39, 39, 40, 40, 41, 41, 41, 42, 42, 42, 86]
ob_val = [4, 5, 2, 4, 18, 14, 79, 2, 2, 16, 2, 1, 3, 1, 1, 38, 3, 1, 1, 3, 1, 66, 1, 1, 1, 1, 2, 1, 4, 6, 175, 1, 1, 2, 5, 2, 1, 214, 1, 8, 5, 2, 7, 1, 4, 32, 1, 4, 15, 24, 30, 3627, 192, 1, 91, 1, 31, 16, 5, 50, 4, 30, 4, 14, 37, 7, 3, 1, 1, 5, 1, 5, 2, 3, 79, 1, 20, 21, 13, 1, 4, 184, 12, 29, 1, 9, 2, 1, 1, 1, 15, 30, 4, 1695, 23, 1, 1, 2, 8, 9, 1, 6, 1, 200, 9, 6, 44, 3, 3, 9, 1218, 2, 1, 17, 2, 5, 1, 4, 14, 29, 11, 84, 260, 16, 34, 59, 1, 1, 1, 1, 1, 93, 1, 8, 5, 15, 11, 7, 1, 1, 1, 1, 154, 71, 2, 7, 4, 134, 2, 2, 40, 5, 1, 4, 3, 39, 37, 1, 2, 2091, 2, 2, 2, 12, 19, 12, 1]
observation = torch.zeros(128, 128)
observation[ob_ref2, ob_ref1] = torch.tensor(ob_val, dtype=observation.dtype)
observation = observation.unsqueeze(0)

t = torch.tensor([1])
alpha_t = torch.e ** (-t)
stationary_sampler_probs = torch.full((128,), 1 / 128)


def get_q_rkm(stationary_sampler_probs, xt):
    xt_one_hot = F.one_hot(xt, len(stationary_sampler_probs))
    q_rkm = alpha_t[:, None] * xt_one_hot + ((1 - alpha_t) * stationary_sampler_probs[xt])[:, None]
    return q_rkm

def get_g_theta_d(stationary_sampler_probs, xt, dim, p_theta_0):
    auxilary_term = 1 + (1 / alpha_t - 1) * stationary_sampler_probs[xt]
    xt_one_hot = F.one_hot(xt, len(stationary_sampler_probs))
    p_theta_d_0 = p_theta_0.sum(dim=dim)
    g_theta_d = (
        (1 - p_theta_d_0[torch.arange(p_theta_d_0.shape[0]), xt] / auxilary_term)[:, None] * stationary_sampler_probs +
        (alpha_t / (1 - alpha_t))[:, None] * p_theta_d_0
    ) * (1 - xt_one_hot) / stationary_sampler_probs[xt][:, None] + xt_one_hot
    return g_theta_d

x1t = torch.tensor([100])
x2t = torch.tensor([27])
y1t = torch.tensor([100])
y2t = torch.tensor([29])
q_rkm_1_x = get_q_rkm(stationary_sampler_probs, x1t)
q_rkm_1_y = q_rkm_1_x
q_rkm_2_x = get_q_rkm(stationary_sampler_probs, x2t)
q_rkm_2_y = get_q_rkm(stationary_sampler_probs, y2t)
q_0_give_t_x = F.normalize(
    (observation * q_rkm_1_x[:, None, :] * q_rkm_2_x[:, :, None]).view(1, -1),
    p=1.0, dim=1
).view(1, 128, 128)
q_0_give_t_y = F.normalize(
    (observation * q_rkm_1_y[:, None, :] * q_rkm_2_y[:, :, None]).view(1, -1),
    p=1.0, dim=1
).view(1, 128, 128)
g_theta_1_x = get_g_theta_d(stationary_sampler_probs, x1t, 1, q_0_give_t_x)
g_theta_2_x = get_g_theta_d(stationary_sampler_probs, x2t, 2, q_0_give_t_x)
g_theta_1_y = get_g_theta_d(stationary_sampler_probs, y1t, 1, q_0_give_t_y)
g_theta_2_y = get_g_theta_d(stationary_sampler_probs, y2t, 2, q_0_give_t_y)

g_theta_1_x_check = []
q_0_give_t_x_1 = q_0_give_t_x.sum(dim=(0, 1))
for y1t_l in range(128):
    tmp = 0
    for x10 in range(128):
        q_t_0_x = F.one_hot(torch.tensor(x10), 128) * alpha_t + (1 - alpha_t) * stationary_sampler_probs
        tmp += q_t_0_x[y1t_l] / q_t_0_x[x1t] * q_0_give_t_x_1[x10]
    g_theta_1_x_check.append(tmp.item())

tilde_R_x_y = stationary_sampler_probs[y2t] / g_theta_2_x[0][y2t]
hat_R_x_y = stationary_sampler_probs[y2t] * g_theta_2_y[0][x2t]

q_t_1 = alpha_t * torch.eye(128) + (1 - alpha_t) * stationary_sampler_probs[None, :]
q_t_2 = q_t_1
q_0 = F.normalize(observation.view(1, -1), p=1.0, dim=1).view(128, 128)
q_t = q_t_2.T @ q_0 @ q_t_1

q_t[y2t, y1t] / q_t[x2t, x1t]
g_theta_2_x[0][y2t]