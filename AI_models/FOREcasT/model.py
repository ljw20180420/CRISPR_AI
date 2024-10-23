from transformers import PretrainedConfig, PreTrainedModel
import torch.nn as nn
import torch
import torch.nn.functional as F

class FOREcasTConfig(PretrainedConfig):
    model_type = "FOREcasT"
    label_names = ["count"]

    def __init__(
        self,
        reg_const = 0.01, # regularization coefficient for insertion
        i1_reg_const = 0.01, # regularization coefficient for deletion
        seed = 63036, # random seed for intialization
        **kwargs
    ):
        self.reg_const = reg_const
        self.i1_reg_const = i1_reg_const
        self.seed = seed
        super().__init__(**kwargs)

class FOREcasTModel(PreTrainedModel):
    config_class = FOREcasTConfig

    @staticmethod
    def get_feature_label():
        def features_pairwise_label(features1_label, features2_label):
            features_label = []
            for label1 in features1_label:
                for label2 in features2_label:
                    features_label.append(f'PW_{label1}_vs_{label2}')
            return features_label
        feature_DelSize_label = ["Any Deletion", "D1", "D2-3", "D4-7", "D8-12", "D>12"]
        feature_InsSize_label = ["Any Insertion", "I1", "I2"]
        feature_DelLoc_label = ['DL-1--1', 'DL-2--2', 'DL-3--3', 'DL-4--6', 'DL-7--10', 'DL-11--15', 'DL-16--30', 'DL<-30', 'DL>=0', 'DR0-0', 'DR1-1', 'DR2-2', 'DR3-5', 'DR6-9', 'DR10-14', 'DR15-29', 'DR<0', 'DR>=30']
        feature_InsSeq_label = ["I1_A", "I1_C", "I1_G", "I1_T", "I2_AA", "I2_AC", "I2_AG", "I2_AT", "I2_CA", "I2_CC", "I2_CG", "I2_CT", "I2_GA", "I2_GC", "I2_GG", "I2_GT", "I2_TA", "I2_TC", "I2_TG", "I2_TT"]
        feature_InsLoc_label = ["IL-1--1", "IL-2--2", "IL-3--3", "IL<-3", "IL>=0"]
        feature_LocalCutSiteSequence_label = []
        for offset in range(-5, 4):
            for nt in ["A", "G", "C", "T"]:
                feature_LocalCutSiteSequence_label.append(f"CS{offset}_NT={nt}")
        feature_LocalCutSiteSeqMatches_label = []
        for offset1 in range(-3, 2):
            for offset2 in range(-3, offset1):
                for nt in ["A", "G", "C", "T"]:
                    feature_LocalCutSiteSeqMatches_label.append(f"M_CS{offset1}_{offset2}_NT={nt}")
        feature_LocalRelativeSequence_label = []
        for offset in range(-3, 3):
            for nt in ["A", "G", "C", "T"]:
                feature_LocalRelativeSequence_label.append(f'L{offset}_NT={nt}')
        for offset in range(-3, 3):
            for nt in ["A", "G", "C", "T"]:
                feature_LocalRelativeSequence_label.append(f'R{offset}_NT={nt}')
        feature_SeqMatches_label = []
        for loffset in range(-3, 3):
            for roffset in range(-3, 3):
                feature_SeqMatches_label.append(f'X_L{loffset}_R{roffset}')
                feature_SeqMatches_label.append(f'M_L{loffset}_R{roffset}')
        feature_I1or2Rpt_label = ['I1Rpt', 'I1NonRpt', 'I2Rpt', 'I2NonRpt']
        feature_microhomology_label = ['L_MH1-1', 'R_MH1-1', 'L_MH2-2', 'R_MH2-2', 'L_MH3-3', 'R_MH3-3', 'L_MM1_MH3-3', 'R_MM1_MH3-3', 'L_MH4-6', 'R_MH4-6', 'L_MM1_MH4-6', 'R_MM1_MH4-6', 'L_MH7-10', 'R_MH7-10', 'L_MM1_MH7-10', 'R_MM1_MH7-10', 'L_MH11-15', 'R_MH11-15', 'L_MM1_MH11-15', 'R_MM1_MH11-15', 'No MH']
        return (
            features_pairwise_label(feature_DelSize_label, feature_DelLoc_label) +
            feature_InsSize_label +
            feature_DelSize_label +
            feature_DelLoc_label +
            feature_InsLoc_label +
            feature_InsSeq_label +
            features_pairwise_label(feature_LocalCutSiteSequence_label, feature_InsSize_label + feature_DelSize_label) +
            features_pairwise_label(feature_microhomology_label + feature_LocalRelativeSequence_label, feature_DelSize_label + feature_DelLoc_label) +
            features_pairwise_label(feature_LocalCutSiteSeqMatches_label + feature_SeqMatches_label, feature_DelSize_label) +
            features_pairwise_label(feature_InsSeq_label + feature_LocalCutSiteSequence_label + feature_LocalCutSiteSeqMatches_label, feature_I1or2Rpt_label) +
            feature_I1or2Rpt_label +
            feature_LocalCutSiteSequence_label +
            feature_LocalCutSiteSeqMatches_label +
            feature_LocalRelativeSequence_label +
            feature_SeqMatches_label +
            feature_microhomology_label
        )

    def __init__(self, config) -> None:
        super().__init__(config)
        # In more recent versions of PyTorch, you no longer need to explicitly register_parameter, it's enough to set a member of your nn.Module with nn.Parameter to "notify" pytorch that this variable should be treated as a trainable parameter (https://stackoverflow.com/questions/59234238/how-to-add-parameters-in-module-class-in-pytorch-custom-model).
        self.generator = torch.Generator().manual_seed(config.seed)
        is_delete = torch.tensor(['I' not in label for label in FOREcasTModel.get_feature_label()])
        self.register_buffer('reg_coff', (is_delete * config.reg_const + ~is_delete * config.i1_reg_const))
        self.linear = nn.Linear(in_features=len(self.reg_coff), out_features=1, bias=False)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1, generator=self.generator)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, feature, count=None) -> torch.Tensor:
        logit = self.linear(feature).squeeze()
        if count is not None:
            return {
                "logit": logit,
                "loss": self.kl_divergence(logit, count)
            }
        return {"logit": logit}

    def kl_divergence(self, logit, count):
        return F.kl_div(
            F.log_softmax(logit, dim=-1),
            F.normalize(count + 0.5, p=1.0, dim=-1), # add 0.5 to prevent log(0), see loadOligoFeaturesAndReadCounts
            reduction='sum'
        ) + logit.shape[0] * (self.reg_coff * (self.linear.weight ** 2)).sum()