import torch.nn as nn
import torch
from config import args
from load_data import train_dataloader

class LogisticRegression(nn.Module):
    def __init__(self, feature_dim, class_dim) -> None:
        super().__init__()
        # In more recent versions of PyTorch, you no longer need to explicitly register_parameter, it's enough to set a member of your nn.Module with nn.Parameter to "notify" pytorch that this variable should be treated as a trainable parameter (https://stackoverflow.com/questions/59234238/how-to-add-parameters-in-module-class-in-pytorch-custom-model).
        self.linear = nn.Linear(in_features=feature_dim, out_features=class_dim)

    def forward(self, x) -> torch.Tensor:
        return self.linear(x)
    
for batch in train_dataloader:
    break

if args.target_model == "indel":
    model = LogisticRegression(batch['input_indel'].shape[1], batch['del_ins_count'].shape[1])
elif args.target_model == "ins":
    model = LogisticRegression(batch['input_ins'].shape[1], batch['ins_count'].shape[1])
elif args.target_model == "del":
    del_model = LogisticRegression(batch['input_del'].shape[1], batch['del_count'].shape[1])