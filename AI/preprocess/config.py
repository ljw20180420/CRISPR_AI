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
        default_config_files=["AI/preprocess/config.yaml"],
        env_prefix="CRISPR_AI",
        # default_env=True,
    )

    parser.add_argument("--config", action="config")

    parser.add_argument(
        "command",
        required=True,
        type=Literal["train", "test", "upload", "inference", "app", "space"],
        help="The command to run. The order is: train -> test -> upload -> inference -> app -> space.",
    )

    parser.add_class_arguments(
        theclass=MyTrain,
        nested_key="train",
    )
    parser.add_method_arguments(
        theclass=MyTrain,
        themethod="get_initializer",
        nested_key="train.initializer",
    )
    parser.add_method_arguments(
        theclass=MyTrain,
        themethod="get_optimizer",
        nested_key="train.optimizer",
    )
    parser.add_method_arguments(
        theclass=MyTrain,
        themethod="get_scheduler",
        nested_key="train.scheduler",
    )

    parser.add_class_arguments(
        theclass=MyTest,
        nested_key="test",
    )

    parser.add_class_arguments(
        theclass=MyGenerator,
        nested_key="generator",
    )
    parser.add_function_arguments(
        function=get_dataset,
        nested_key="dataset",
        skip=["my_generator"],
    )
    parser.add_function_arguments(
        function=get_metrics,
        nested_key="metric",
        skip=["meta_data"],
    )
    parser.add_function_arguments(
        function=get_model,
        nested_key="model",
        skip=["meta_data"],
    )
    parser.add_function_arguments(
        function=get_logger,
        nested_key="logger",
    )
    for metric in metrics:
        parser.add_class_arguments(
            theclass=getattr(
                importlib.import_module(f"AI.preprocess.metric"),
                metric,
            ),
            nested_key=f"metric.{metric}",
        )

    for preprocess, model_types in preprocess_to_model.items():
        for model_type in model_types:
            parser.add_class_arguments(
                theclass=getattr(
                    importlib.import_module(f"AI.preprocess.{preprocess}.model"),
                    f"{model_type}Config",
                ),
                nested_key=f"model.{preprocess}.{model_type}",
                skip=["**kwargs"],
            )

    return parser.parse_args().as_dict()
