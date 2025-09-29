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

    train_parser.add_argument(
        "--metric",
        nargs="+",
        type=CrossEntropyBase,
        required=True,
        enable_path=True,
    )

    train_parser.add_subclass_arguments(
        baseclass=(
            CRIformer_M.CRIformer,
            CRIfuser_M.CRIfuser,
            DeepHF_M.DeepHF,
            DeepHF_M.MLP,
            DeepHF_M.CNN,
            DeepHF_M.XGBoost,
            DeepHF_M.SGDClassifier,
            FOREcasT_M.FOREcasT,
            inDelphi_M.inDelphi,
            Lindel_M.Lindel,
        ),
        nested_key="model",
    )

    return parser, train_parser, test_parser
