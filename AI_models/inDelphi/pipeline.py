from diffusers import DiffusionPipeline, __version__
import torch
from sklearn.neighbors import KNeighborsRegressor

class inDelphiPipeline(DiffusionPipeline):
    def __init__(self, inDelphi_model):
        super().__init__()

        self.register_modules(inDelphi_model=inDelphi_model)
        self.inDelphi_model.insertion_model = KNeighborsRegressor(weights='distance').fit((inDelphi_model.onebp_features.numpy() - inDelphi_model.onebp_feature_mean.numpy()) / inDelphi_model.onebp_feature_std.numpy(), inDelphi_model.insert_probabilities)

    @torch.no_grad()
    def __call__(self, batch, m654=False):
        mh_weights, mhless_weights, total_del_len_weights = self.inDelphi_model(batch["mh_input"], batch["mh_del_len"]).values()
        mX = self.inDelphi_model.m654 if m654 else self.inDelphi_model.m4
        log_total_weights = total_del_len_weights.sum(dim=1, keepdim=True).log()
        precisions = 1 - torch.distributions.Categorical(total_del_len_weights[:,:28]).entropy() / torch.log(torch.tensor(28))
        onebp_features = torch.cat([
            batch["onebp_feature"],
            precisions[:, None],
            log_total_weights
        ], dim=1).numpy()
        pre_insert_probabilities = self.inDelphi_model.insertion_model.predict((onebp_features - self.inDelphi_model.onebp_feature_mean.numpy()) / self.inDelphi_model.onebp_feature_std.numpy())
        pre_insert_1bps = mX[batch['m654'] // 25] if mX.shape[0] == 5 else mX[batch['m654']]
        return {
            "mh_gt_pos": batch["mh_gt_pos"],
            "mh_del_len": [
                batch["mh_del_len"][i][:len(batch["mh_gt_pos"][i])].tolist()
                for i in range(len(batch["mh_gt_pos"]))
            ],
            "mh_mh_len": [
                batch["mh_input"][i, :, 0][:len(batch["mh_gt_pos"][i])].to(torch.int16).tolist()
                for i in range(len(batch["mh_gt_pos"]))
            ],
            "mh_weight": [
                mh_weights[i, :len(batch["mh_gt_pos"][i])].tolist()
                for i in range(len(batch["mh_gt_pos"]))
            ],
            "mhless_weight": mhless_weights,
            "total_del_len_weight": total_del_len_weights,
            "pre_insert_probability": pre_insert_probabilities,
            "pre_insert_1bp": pre_insert_1bps
        }