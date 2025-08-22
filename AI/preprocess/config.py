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
            CRIformer_M.CRIformerConfig,
            CRIfuser_M.CRIfuserConfig,
            DeepHF_M.DeepHFConfig,
            DeepHF_M.MLPConfig,
            DeepHF_M.CNNConfig,
            DeepHF_M.XGBoostConfig,
            DeepHF_M.RidgeConfig,
            FOREcasT_M.FOREcasTConfig,
            inDelphi_M.inDelphiConfig,
            Lindel_M.LindelConfig,
        ),
        nested_key="model",
    )

    return parser, train_parser, test_parser
