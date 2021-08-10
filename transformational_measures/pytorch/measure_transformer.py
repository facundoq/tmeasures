import torch
import abc

from . import PyTorchMeasureResult


class MeasureTransformation(torch.nn.Module):

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, activation_name: str) -> torch.Tensor:
        pass

    def transform_result(self, r: PyTorchMeasureResult):
        for i, (l, n) in enumerate(zip(r.layers, r.layer_names)):
            r.layers[i] = self.forward(l, n)

    def __repr__(self):
        return self.__class__.__name__


class NoTransformation(MeasureTransformation):

    def forward(self, x: torch.Tensor, activation_name: str) -> torch.Tensor:
        return x


class AverageExtraDimensions(MeasureTransformation):

    def forward(self, x: torch.Tensor, activation_name: str) -> torch.Tensor:
        dims = tuple(range(1, len(x.shape)))
        return x.mean(dim=dims)


class AverageFeatureMaps(MeasureTransformation):
    def __init__(self, channel_dimension: int = 0):
        super().__init__()
        self.channel_dimension = channel_dimension

    def forward(self, x: torch.Tensor, activation_name: str) -> torch.Tensor:
        d = len(x.shape)
        if d != 3:
            # ignore non-feature map activations
            return x

        dims = tuple(filter(lambda n: n != self.channel_dimension, range(d)))

        return x.mean(dim=dims)

    def __repr__(self):
        extra = ""
        if self.channel_dimension!=0:
            extra = f"(dim={self.channel_dimension})"
        return f"{super().__repr__()}{extra}"
