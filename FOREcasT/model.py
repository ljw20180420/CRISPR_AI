import torch.nn as nn
import torch
from config import args

class LogisticRegression(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # In more recent versions of PyTorch, you no longer need to explicitly register_parameter, it's enough to set a member of your nn.Module with nn.Parameter to "notify" pytorch that this variable should be treated as a trainable parameter (https://stackoverflow.com/questions/59234238/how-to-add-parameters-in-module-class-in-pytorch-custom-model).
        self.register_buffer('reg_coff', self.get_reg_coff())
        self.theta = nn.Parameter(torch.zeros(len(self.reg_coff)))

    def forward(self, x) -> torch.Tensor:
        return (x * self.theta).sum(dim=2)
    
    def get_reg_coff(self) -> torch.Tensor:
        tmp_feature_file = (args.data_file.parent / "FOREcasT/tmp/tmp_features_0.txt").as_posix()
        with open(tmp_feature_file, 'r') as fd:
            _ = fd.readline()
            _ = fd.readline()
            feature_columns = fd.readline().rstrip('\n').split('\t')[4:]
        is_delete = torch.tensor(['I' not in feature for feature in feature_columns])
        return (is_delete * args.reg_const + ~is_delete * args.i1_reg_const)
    
model = LogisticRegression()