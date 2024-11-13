#!/usr/bin/env python

import torch
import torch.nn.functional as F
from .model import CRISPRDiffuserConfig, CRISPRDiffuserModel
from ..config import get_config, get_logger

args = get_config("config_CRISPR_diffuser.ini")
logger = get_logger(args)

class PerfectModel(CRISPRDiffuserModel):
    config_class = CRISPRDiffuserConfig

    def __init__(self, config):
        super().__init__(config)
        ob_ref1 = [110, 103, 96, 103, 100, 98, 106, 105, 96, 101, 105, 109, 113, 90, 95, 99, 102, 111, 95, 96, 99, 100, 101, 113, 86, 92, 94, 95, 99, 100, 101, 102, 103, 107, 89, 92, 98, 100, 101, 114, 81, 85, 89, 91, 92, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 110, 91, 92, 93, 94, 95, 96, 97, 99, 100, 101, 102, 103, 104, 106, 110, 123, 87, 88, 89, 91, 93, 94, 95, 97, 98, 99, 100, 101, 103, 83, 91, 92, 93, 94, 95, 96, 99, 100, 102, 87, 88, 90, 91, 93, 95, 96, 97, 98, 99, 100, 101, 103, 105, 87, 90, 91, 98, 99, 100, 86, 96, 99, 100, 101, 103, 91, 94, 99, 100, 101, 103, 86, 90, 92, 94, 100, 101, 88, 90, 91, 92, 95, 98, 90, 93, 94, 95, 97, 99, 100, 102, 89, 91, 94, 95, 96, 97, 100, 97, 99, 100, 101, 97, 99, 93, 99, 100, 97, 100, 102, 105]
        ob_ref2 = [2, 5, 13, 13, 15, 16, 18, 19, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 34, 34, 34, 34, 34, 34, 35, 35, 35, 35, 35, 35, 36, 36, 36, 36, 36, 36, 37, 37, 37, 37, 37, 37, 37, 37, 38, 38, 38, 38, 38, 38, 38, 39, 39, 39, 39, 40, 40, 41, 41, 41, 42, 42, 42, 86]
        ob_val = [4, 5, 2, 4, 18, 14, 79, 2, 2, 16, 2, 1, 3, 1, 1, 38, 3, 1, 1, 3, 1, 66, 1, 1, 1, 1, 2, 1, 4, 6, 175, 1, 1, 2, 5, 2, 1, 214, 1, 8, 5, 2, 7, 1, 4, 32, 1, 4, 15, 24, 30, 3627, 192, 1, 91, 1, 31, 16, 5, 50, 4, 30, 4, 14, 37, 7, 3, 1, 1, 5, 1, 5, 2, 3, 79, 1, 20, 21, 13, 1, 4, 184, 12, 29, 1, 9, 2, 1, 1, 1, 15, 30, 4, 1695, 23, 1, 1, 2, 8, 9, 1, 6, 1, 200, 9, 6, 44, 3, 3, 9, 1218, 2, 1, 17, 2, 5, 1, 4, 14, 29, 11, 84, 260, 16, 34, 59, 1, 1, 1, 1, 1, 93, 1, 8, 5, 15, 11, 7, 1, 1, 1, 1, 154, 71, 2, 7, 4, 134, 2, 2, 40, 5, 1, 4, 3, 39, 37, 1, 2, 2091, 2, 2, 2, 12, 19, 12, 1]
        self.register_buffer("observation", torch.zeros(config.ref2len + 1, config.ref1len + 1))
        self.observation[ob_ref2, ob_ref1] = torch.tensor(ob_val, dtype=self.observation.dtype)

    def forward(self, x1t_x2t_t: dict, condition: torch.Tensor, observation: torch.Tensor | None = None):
        x1t, x2t, t = x1t_x2t_t.values()
        alpha_t = torch.e ** (-t)
        x1t_one_hot = F.one_hot(x1t, len(self.stationary_sampler1_probs))
        x2t_one_hot = F.one_hot(x2t, len(self.stationary_sampler2_probs))
        q_rkm_1 = alpha_t[:, None] * x1t_one_hot + ((1 - alpha_t) * self.stationary_sampler1_probs[x1t])[:, None]
        q_rkm_2 = alpha_t[:, None] * x2t_one_hot + ((1 - alpha_t) * self.stationary_sampler2_probs[x2t])[:, None]
        return {
             "p_theta_0_logit": (self.observation * q_rkm_1[:, None, :] * q_rkm_2[:, :, None]).log().clamp_min(-1000)
        }
