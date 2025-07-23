import jsonargparse
import importlib
import importlib
from .train import MyTrain
from .generator import MyGenerator
from .dataset import MyDataset
from .initializer import MyInitializer
from .optimizer import MyOptimizer
from .lr_scheduler import MyLRScheduler
from .utils import MyLogger

preprocess_to_model = {
    "inDelphi": ["inDelphi"],
    "FOREcasT": ["FOREcasT"],
    "Lindel": ["Lindel"],
    "DeepHF": ["DeepHF"],
    "CRIformer": ["CRIformer"],
    "CRIfuser": ["CRIfuser"],
}

metrics = ["NonWildTypeCrossEntropy"]


def get_config() -> jsonargparse.Namespace:
    parser = jsonargparse.ArgumentParser(
        description="Arguments of AI models.",
        default_config_files=["AI/preprocess/config.yaml"],
        env_prefix="CRISPR_AI",
        # default_env=True,
    )

    parser.add_argument("--config", action="config")

    parser.add_class_arguments(
        theclass=MyTrain,
        nested_key="train",
    )

    parser.add_class_arguments(
        theclass=MyGenerator,
        nested_key="generator",
    )
    parser.add_class_arguments(
        theclass=MyDataset,
        nested_key="dataset",
        skip=["my_generator"],
    )
    parser.add_class_arguments(
        theclass=MyInitializer,
        nested_key="initializer",
    )
    parser.add_class_arguments(
        theclass=MyOptimizer,
        nested_key="optimizer",
        skip=["model"],
    )
    parser.add_class_arguments(
        theclass=MyLRScheduler,
        nested_key="lr_scheduler",
        skip=["num_training_steps", "my_optimizer"],
    )
    parser.add_class_arguments(
        theclass=MyLogger,
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
                nested_key=f"{preprocess}.{model_type}",
                skip=["**kwargs"],
            )

    return parser.parse_args()
