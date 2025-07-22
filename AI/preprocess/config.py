import jsonargparse
import pathlib
import importlib
from dataclasses import dataclass
from typing import Literal
import importlib
from .generator import MyGenerator
from .initializer import MyInitializer
from .dataset import MyDataset
from .optimizer import MyOptimizer
from .lr_scheduler import MyLRScheduler


@dataclass
class Common:
    """Common parameters.

    Args:
        trial_name: name of the training trial
        output_dir: Output directory.
        batch_size: Batch size.
        num_epochs: Total number of training epochs to perform.
        device: Device.
        log_level: Logging level.
        inference_data: The data file of inference.
        inference_output: The output file of inference.
    """

    trial_name: str
    output_dir: pathlib.Path
    batch_size: int
    num_epochs: int
    device: Literal["cpu", "cuda"]
    log_level: Literal[
        "CRITICAL", "FATAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"
    ]
    inference_data: pathlib.Path
    inference_output: pathlib.Path


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
        default_config_files=["preprocess/config.yaml"],
        env_prefix="CRISPR_AI",
        # default_env=True,
    )

    parser.add_argument("--config", action="config")

    parser.add_argument(
        "command",
        type=Literal["train", "test", "upload", "inference", "app", "space"],
        help="The command. The run order is: train -> test -> upload -> inference -> app -> space.",
    )

    # Add global arguments.
    parser.add_class_arguments(theclass=Common)
    parser.add_class_arguments(theclass=MyGenerator, nested_key="generator")
    parser.add_class_arguments(theclass=MyInitializer, nested_key="initializer")
    parser.add_class_arguments(theclass=MyDataset, nested_key="dataset")
    parser.add_class_arguments(theclass=MyOptimizer, nested_key="optimizer")
    parser.add_class_arguments(theclass=MyLRScheduler, nested_key="lr_scheduler")
    for metric in metrics:
        parser.add_class_arguments(
            theclass=getattr(
                importlib.import_module(f"preprocess.metric"),
                metric,
            ),
            nested_key=f"metric.{metric}",
        )

    parser_subcommands = parser.add_subcommands(required=False, dest="preprocess")
    for preprocess, model_names in preprocess_to_model.items():
        preprocess_command = jsonargparse.ArgumentParser(
            description=f"preprocess {preprocess}.",
        )
        parser_subcommands.add_subcommand(preprocess, preprocess_command)
        preprocess_subcommands = preprocess_command.add_subcommands(
            required=False, dest="model_name"
        )
        for model_name in model_names:
            model_command = jsonargparse.ArgumentParser(
                description=f"model {model_name} of preprocess {preprocess}.",
                default_config_files=[
                    f"preprocess/{preprocess}/configs/{model_name}.yaml"
                ],
            )
            preprocess_subcommands.add_subcommand(model_name, model_command)
            model_command.add_argument("--config", action="config")
            model_command.add_class_arguments(
                theclass=getattr(
                    importlib.import_module(f"preprocess.{preprocess}.model"),
                    f"{model_name}Config",
                ),
            )

    return parser.parse_args()
