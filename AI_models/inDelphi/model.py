from transformers import PretrainedConfig, PreTrainedModel
import torch
import torch.nn.functional as F
import torch.nn as nn

class inDelphiConfig(PretrainedConfig):
    model_type = "inDelphi"
    label_names = ["genotype_count", "total_del_len_count"]

    def __init__(
        self,
        DELLEN_LIMIT = 60, # the upper limit of deletion length (strictly less than DELLEN_LIMIT)
        mid_dim = 16, # the size of middle layer of MLP
        seed = 63036, # random seed for intialization
        **kwargs,
    ):
        self.DELLEN_LIMIT = DELLEN_LIMIT
        self.mid_dim = mid_dim
        self.seed = seed
        super().__init__(**kwargs)

class inDelphiModel(PreTrainedModel):
    config_class = inDelphiConfig

    def __init__(self, config):
        super().__init__(config)
        # In more recent versions of PyTorch, you no longer need to explicitly register_parameter, it's enough to set a member of your nn.Module with nn.Parameter to "notify" pytorch that this variable should be treated as a trainable parameter (https://stackoverflow.com/questions/59234238/how-to-add-parameters-in-module-class-in-pytorch-custom-model).
        self.generator = torch.Generator().manual_seed(config.seed)
        self.DELLEN_LIMIT = config.DELLEN_LIMIT
        self.register_buffer('del_lens', torch.arange(1, config.DELLEN_LIMIT, dtype=torch.float32))
        self.mh_in_layer = nn.Linear(in_features=2, out_features=config.mid_dim)
        self.mh_mid_layer = nn.Linear(in_features=config.mid_dim, out_features=config.mid_dim)
        self.mh_out_layer = nn.Linear(in_features=config.mid_dim, out_features=1)
        self.mhless_in_layer = nn.Linear(in_features=1, out_features=config.mid_dim)
        self.mhless_mid_layer = nn.Linear(in_features=config.mid_dim, out_features=config.mid_dim)
        self.mhless_out_layer = nn.Linear(in_features=config.mid_dim, out_features=1)
        self.mid_active = self.sigmoid
        self.out_active = self.logit_to_weight
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1, generator=self.generator)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, mh_input, mh_del_len, genotype_count=None, total_del_len_count=None):
        batch_size = mh_input.shape[0]
        mh_weight = self.mh_in_layer(mh_input)
        mh_weight = self.mid_active(mh_weight)
        mh_weight = self.mh_mid_layer(mh_weight)
        mh_weight = self.mid_active(mh_weight)
        mh_weight = self.mh_out_layer(mh_weight)
        mh_weight = self.out_active(mh_weight, mh_del_len)

        mhless_weight = self.mhless_in_layer(self.del_lens[:, None])
        mhless_weight = self.mid_active(mhless_weight)
        mhless_weight = self.mhless_mid_layer(mhless_weight)
        mhless_weight = self.mid_active(mhless_weight)
        mhless_weight = self.mhless_out_layer(mhless_weight)
        mhless_weight = self.out_active(mhless_weight, self.del_lens)

        total_del_len_weight = torch.zeros(batch_size, mhless_weight.shape[0] + 1, dtype=mh_weight.dtype, device=mh_weight.device).scatter_add(dim=1, index=mh_del_len - 1, src=mh_weight)[:, :-1] + mhless_weight
        if genotype_count is not None and total_del_len_count is not None:
            loss = self.negative_correlation(mh_weight, mhless_weight, total_del_len_weight, genotype_count, total_del_len_count)
            return {
                "mh_weight": mh_weight,
                "mhless_weight": mhless_weight,
                "total_del_len_weight": total_del_len_weight,
                "loss": loss
            }
        return {
            "mh_weight": mh_weight,
            "mhless_weight": mhless_weight,
            "total_del_len_weight": total_del_len_weight
        }

    def logit_to_weight(self, logits, del_lens):
        return torch.exp(logits.squeeze() - 0.25 * del_lens) * (del_lens < self.DELLEN_LIMIT)

    def sigmoid(self, x):
        return 0.5 * (F.tanh(x) + 1)

    def negative_correlation(self, mh_weight, mhless_weight, total_del_len_weight, genotype_count, total_del_len_count):
        batch_size = mh_weight.shape[0]
        genotype_pearson = (
            F.normalize(
                torch.cat(
                    (mh_weight, mhless_weight.expand(batch_size, -1)),
                    dim = 1
                ),
                p=2.0,
                dim=1
            ) *
            F.normalize(genotype_count, p=2.0, dim=1)
        ).sum()

        total_del_len_pearson = (
            F.normalize(total_del_len_weight, p=2.0, dim=1) *
            F.normalize(total_del_len_count, p=2.0, dim=1)
        ).sum()
        
        return -genotype_pearson - total_del_len_pearson