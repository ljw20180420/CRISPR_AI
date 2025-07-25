from transformers import PreTrainedModel
import importlib


def get_model(
    preprocess: str,
    model_type: str,
    meta_data: dict,
) -> PreTrainedModel:
    """Model arguments.

    Args:
        preprocess: The preprocess.
        model_type: The model type.
    """
    model_module = importlib.import_module(f"AI.preprocess.{preprocess}.model")
    return getattr(model_module, f"{model_type}Model")(
        getattr(model_module, f"{model_type}Config")(
            **meta_data[preprocess][model_type],
        )
    )
