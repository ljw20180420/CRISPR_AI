from diffusers import DiffusionPipeline, __version__
import torch
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

class inDelphiPipeline(DiffusionPipeline):
    def __init__(self, inDelphi_model, onebp_features, insert_probabilities, m654):
        super().__init__()

        self.register_modules(inDelphi_model=inDelphi_model)
        self.onebp_feature_mean = onebp_features.mean(axis=0)
        self.onebp_feature_std = onebp_features.std(axis=0)
        self.insertion_model = KNeighborsRegressor(weights='distance').fit((onebp_features - self.onebp_feature_mean) / self.onebp_feature_std, insert_probabilities)
        self.m654 = m654 / np.maximum(np.linalg.norm(m654, ord=1, axis=1, keepdims=True), 1e-6)
        self.m4 = m654.reshape(5, 25, 5).sum(axis=1)
        self.m4 = self.m4 / np.maximum(np.linalg.norm(self.m4, ord=1, axis=1, keepdims=True), 1e-6)

    @torch.no_grad()
    def __call__(self, batch, use_m654=False):
        mh_weights, mhless_weights, total_del_len_weights = self.inDelphi_model(
            batch["mh_input"].to(self.inDelphi_model.device),
            batch["mh_del_len"].to(self.inDelphi_model.device)
        ).values()
        mX = self.m654 if use_m654 else self.m4
        log_total_weights = total_del_len_weights.sum(dim=1, keepdim=True).log()
        precisions = 1 - torch.distributions.Categorical(total_del_len_weights[:,:28]).entropy() / torch.log(torch.tensor(28))
        onebp_features = torch.cat([
            batch["onebp_feature"],
            precisions[:, None].cpu(),
            log_total_weights.cpu()
        ], dim=1).cpu().numpy()
        pre_insert_probabilities = self.insertion_model.predict((onebp_features - self.onebp_feature_mean) / self.onebp_feature_std)
        pre_insert_1bps = mX[batch['m654'] // 25] if mX.shape[0] == 5 else mX[batch['m654']]
        return {
            "mh_weight": [
                mh_weights[i, batch["mh_del_len"][i] < self.inDelphi_model.config.DELLEN_LIMIT]
                for i in range(len(batch["mh_del_len"]))
            ],
            "mhless_weight": mhless_weights,
            "total_del_len_weight": total_del_len_weights,
            "pre_insert_probability": pre_insert_probabilities,
            "pre_insert_1bp": pre_insert_1bps
        }