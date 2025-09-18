import jsonargparse
from .CRIformer import model as CRIformer_M
from .CRIfuser import model as CRIfuser_M
from .DeepHF import model as DeepHF_M
from .FOREcasT import model as FOREcasT_M
from .inDelphi import model as inDelphi_M
from .Lindel import model as Lindel_M
from .metric import CrossEntropyBase
from .dataset import get_dataset
from common_ai import config


def get_config() -> tuple[jsonargparse.ArgumentParser]:
    parser, train_parser, test_parser = config.get_config()

    train_parser.add_function_arguments(
        function=get_dataset,
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
