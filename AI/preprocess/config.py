import jsonargparse
from .CRIformer.model import CRIformerConfig
from .CRIfuser.model import CRIfuserConfig
from .DeepHF.model import DeepHFConfig
from .FOREcasT.model import FOREcasTConfig
from .inDelphi.model import inDelphiConfig
from .Lindel.model import LindelConfig
from .train import MyTrain
from .test import MyTest
from .metric import NonWildTypeCrossEntropy
from .dataset import get_dataset
from .utils import MyGenerator, get_logger


def get_config() -> tuple[jsonargparse.ArgumentParser]:
    parser = jsonargparse.ArgumentParser(
        description="Arguments of AI models.",
    )
    subcommands = parser.add_subcommands(required=True, dest="subcommand")

    test_parser = jsonargparse.ArgumentParser(description="Test AI models.")
    test_parser.add_argument("--config", action="config")
    test_parser.add_class_arguments(theclass=MyTest, nested_key="test")
    subcommands.add_subcommand(name="test", parser=test_parser)

    train_parser = jsonargparse.ArgumentParser(description="Train AI models.")
    train_parser.add_argument("--config", action="config")
    train_parser.add_class_arguments(theclass=MyTrain, nested_key="train")
    train_parser.add_method_arguments(
        theclass=MyTrain,
        themethod="get_initializer",
        nested_key="initializer",
    )
    train_parser.add_method_arguments(
        theclass=MyTrain,
        themethod="get_optimizer",
        nested_key="optimizer",
    )
    train_parser.add_method_arguments(
        theclass=MyTrain,
        themethod="get_lr_scheduler",
        nested_key="lr_scheduler",
    )

    train_parser.add_class_arguments(
        theclass=MyGenerator,
        nested_key="generator",
    )
    train_parser.add_function_arguments(
        function=get_logger,
        nested_key="logger",
    )

    train_parser.add_function_arguments(
        function=get_dataset,
        nested_key="dataset",
    )

    train_parser.add_argument(
        "--metric",
        nargs="+",
        type=NonWildTypeCrossEntropy,
        required=True,
        enable_path=True,
    )

    train_parser.add_subclass_arguments(
        baseclass=(
            CRIformerConfig,
            CRIfuserConfig,
            DeepHFConfig,
            FOREcasTConfig,
            inDelphiConfig,
            LindelConfig,
        ),
        nested_key="model",
    )

    subcommands.add_subcommand(name="train", parser=train_parser)

    return parser, train_parser
