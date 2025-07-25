import jsonargparse
import importlib
from typing import Literal
from .train import MyTrain
from .test import MyTest
from .metric import get_metrics
from .model import get_model
from .dataset import get_dataset
from .utils import MyGenerator, get_logger

preprocess_to_model = {
    "inDelphi": ["inDelphi"],
    "FOREcasT": ["FOREcasT"],
    "Lindel": ["Lindel"],
    "DeepHF": ["DeepHF"],
    "CRIformer": ["CRIformer"],
    "CRIfuser": ["CRIfuser"],
}

metrics = ["NonWildTypeCrossEntropy"]


def get_config() -> jsonargparse.ArgumentParser:
    parser = jsonargparse.ArgumentParser(
        description="Arguments of AI models.",
    )
    subcommands = parser.add_subcommands(required=True, dest="subcommand")

    test_parser = jsonargparse.ArgumentParser(
        description="Test AI models.",
        default_config_files=["AI/preprocess/test.yaml"],
    )
    test_parser.add_class_arguments(theclass=MyTest, nested_key="test")
    test_parser.add_argument("--config", action="config")
    subcommands.add_subcommand(name="test", parser=test_parser)

    train_parser = jsonargparse.ArgumentParser(
        description="Train AI models.",
        default_config_files=["AI/preprocess/train.yaml"],
    )
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
        skip=["my_generator"],
    )

    train_parser.add_function_arguments(
        function=get_metrics,
        nested_key="metric",
        skip=["meta_data"],
    )
    for metric in metrics:
        train_parser.add_class_arguments(
            theclass=getattr(
                importlib.import_module(f"AI.preprocess.metric"),
                metric,
            ),
            nested_key=metric,
        )

    train_parser.add_function_arguments(
        function=get_model,
        nested_key="model",
        skip=["meta_data"],
    )
    for preprocess, model_types in preprocess_to_model.items():
        for model_type in model_types:
            train_parser.add_class_arguments(
                theclass=getattr(
                    importlib.import_module(f"AI.preprocess.{preprocess}.model"),
                    f"{model_type}Config",
                ),
                nested_key=f"{preprocess}.{model_type}",
                skip=["**kwargs"],
            )

    subcommands.add_subcommand(name="train", parser=train_parser)

    return parser.parse_args().as_dict()
