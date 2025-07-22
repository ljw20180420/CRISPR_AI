import os
import torch
import safetensors.torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig
from typing import Literal
from .generator import MyGenerator


class BaseConfig(PretrainedConfig):
    def __init__(
        self,
        my_generator: MyGenerator,
        initializer: Literal["lecun_uniform", "normal", "he_normal", "he_uniform"],
        **kwargs,
    ):
        """Basic model arguments.

        Args:
            my_generator: Random generator.
            initializer: Intialization methods for model weights.
        """
        self.my_generator = my_generator
        self.initializer = initializer
        super().__init__(**kwargs)


class BaseModel(PreTrainedModel):
    def __init__(self, config: BaseConfig) -> None:
        super().__init__(config)

    def _auto_set_generator(self) -> None:
        if self.device == torch.device("cpu"):
            self.generator = self.config.my_generator.torch_c_rng
        else:
            self.generator = self.config.my_generator.torch_g_rng

    # huggingface use the name initialize_weights, use another name here.
    def _initialize_model_layer_weights(self) -> None:
        self._auto_set_generator()

        if self.config.initializer == "lecun_uniform":
            init_func = (
                lambda weight, generator=self.generator: nn.init.kaiming_uniform_(
                    weight, nonlinearity="linear", generator=generator
                )
            )
        elif self.config.initializer == "normal":
            init_func = lambda weight, generator=self.generator: nn.init.normal_(
                weight, generator=generator
            )
        elif self.config.initializer == "he_normal":
            init_func = nn.init.kaiming_normal_
        elif self.config.initializer == "he_uniform":
            init_func = nn.init.kaiming_uniform_

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv2d):
                init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.ConvTranspose2d):
                init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def load_checkpoint(self, checkpoint_path: os.PathLike):
        state_dict = safetensors.torch.load_file(
            os.path.join(
                os.fspath(checkpoint_path),
                "model.safetensors",
            ),
            device="cpu",
        )
        load_result = self.load_state_dict(state_dict, False)
        # release memory
        del state_dict
        if len(load_result.missing_keys) != 0:
            self.logger.warning(
                f"There were missing keys in the checkpoint model loaded: {load_result.missing_keys}."
            )
        if len(load_result.unexpected_keys) != 0:
            self.logger.warning(
                f"There were unexpected keys in the checkpoint model loaded: {load_result.unexpected_keys}."
            )
