from transformers import PretrainedConfig, PreTrainedModel
import torch.nn as nn
import torch
import torch.nn.functional as F

from transformers import PretrainedConfig

class LindelConfig(PretrainedConfig):
    model_type = "Lindel"
    label_names = ["count"]

    def __init__(
        self,
        dlen = 30, # the upper limit of deletion length (strictly less than dlen)
        mh_len = 4, # the upper limit of micro-homology length
        model = "indel", # the actual model, should be "indel", "del", or "ins"
        reg_mode = "l2", # regularization method, should be "l2" or "l1"
        reg_const = 0.01, # regularization coefficient
        seed = 63036, # random seed for intialization
        **kwargs,
    ):
        self.dlen = dlen
        self.mh_len = mh_len
        self.model = model
        self.reg_mode = reg_mode
        self.reg_const = reg_const
        self.seed = seed
        super().__init__(**kwargs)

class LindelModel(PreTrainedModel):
    config_class = LindelConfig

    def __init__(self, config) -> None:
        super().__init__(config)
        # In more recent versions of PyTorch, you no longer need to explicitly register_parameter, it's enough to set a member of your nn.Module with nn.Parameter to "notify" pytorch that this variable should be treated as a trainable parameter (https://stackoverflow.com/questions/59234238/how-to-add-parameters-in-module-class-in-pytorch-custom-model).
        self.generator = torch.Generator().manual_seed(config.seed)
        self.reg_mode = config.reg_mode
        self.reg_const = config.reg_const
        if config.model == "indel":
            feature_dim = 20 * 4 + 19 * 16
            class_dim = 2
        elif config.model == "ins":
            feature_dim = 6 * 4 + 5 * 16
            class_dim = 21
        elif config.model == "del":
            class_dim = (4 + 1 + 4 + config.dlen - 1) * (config.dlen - 1) // 2
            feature_dim = class_dim * (config.mh_len + 1) + 20 * 4 + 19 * 16
        self.linear = nn.Linear(in_features=feature_dim, out_features=class_dim)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1, generator=self.generator)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input, count=None) -> torch.Tensor:
        logit = self.linear(input)
        if count is not None:
            return {
                "logit": logit,
                "loss": self.cross_entropy_reg(logit, count)
            }
        return {"logit": logit}

    def cross_entropy_reg(self, logit, count):
        if self.reg_mode == "l2":
            reg_term = (self.linear.weight ** 2).sum()
        elif self.reg_mode == "l1":
            reg_term = abs(self.linear.weight).sum()
        return -(F.log_softmax(logit, dim=1) * F.normalize(count.to(torch.float32), p=1.0, dim=1)).sum() + logit.shape[0] * self.reg_const * reg_term