import jsonargparse
from .preprocess.CRIformer import model as CRIformer_M
from .preprocess.CRIfuser import model as CRIfuser_M
from .preprocess.DeepHF import model as DeepHF_M
from .preprocess.FOREcasT import model as FOREcasT_M
from .preprocess.inDelphi import model as inDelphi_M
from .preprocess.Lindel import model as Lindel_M
from .metric import CrossEntropyBase
from .dataset import MyDataset
from common_ai import config


def get_config() -> tuple[jsonargparse.ArgumentParser]:
    parser, train_parser, test_parser = config.get_config()

    train_parser.add_class_arguments(
        theclass=MyDataset,
        nested_key="dataset",
    )

    train_parser.add_argument(
        "--metric",
        nargs="+",
        type=CrossEntropyBase,
        required=True,
        enable_path=True,
    )

    train_parser.add_subclass_arguments(
        baseclass=(
            CRIformer_M.CRIformerModel,
            CRIfuser_M.CRIfuserModel,
            DeepHF_M.DeepHFModel,
            DeepHF_M.MLPModel,
            DeepHF_M.CNNModel,
            DeepHF_M.XGBoostModel,
            DeepHF_M.SGDClassifierModel,
            FOREcasT_M.FOREcasTModel,
            inDelphi_M.inDelphiModel,
            Lindel_M.LindelModel,
        ),
        nested_key="model",
    )

    return parser, train_parser, test_parser
