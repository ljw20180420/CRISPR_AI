from typing import Literal
from torch import nn
from transformers import PreTrainedModel
from .generator import MyGenerator


class MyInitializer:
    def __init__(
        self,
        name: Literal["lecun_uniform", "normal", "he_normal", "he_uniform"],
    ) -> None:
        """Initializer arguments.

        Args:
            name: Name of the intialization method for model weights.
        """
        self.name = name

    def __call__(self, model: PreTrainedModel, my_generator: MyGenerator) -> None:
        generator = my_generator.get_torch_generator_by_device(model.device)

        if self.name == "lecun_uniform":
            init_func = lambda weight, generator=generator: nn.init.kaiming_uniform_(
                weight, nonlinearity="linear", generator=generator
            )
        elif self.name == "normal":
            init_func = lambda weight, generator=generator: nn.init.normal_(
                weight, generator=generator
            )
        elif self.name == "he_normal":
            init_func = lambda weight, generator=generator: nn.init.kaiming_normal_(
                weight, generator=generator
            )
        elif self.name == "he_uniform":
            init_func = lambda weight, generator=generator: nn.init.kaiming_uniform_(
                weight, generator=generator
            )

        for m in model.modules():
            # linear layers
            if isinstance(m, nn.Linear) or isinstance(m, nn.Bilinear):
                init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # (transposed) convolution layers
            if (
                isinstance(m, nn.Conv1d)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.Conv3d)
                or isinstance(m, nn.ConvTranspose1d)
                or isinstance(m, nn.ConvTranspose2d)
                or isinstance(m, nn.ConvTranspose3d)
            ):
                init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
