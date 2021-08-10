import abc
from typing import Iterable

import torch

from transformational_measures import Transformation, TransformationSet


class PyTorchTransformation(Transformation,torch.nn.Module):

    @abc.abstractmethod
    def parameters(self)->torch.Tensor:
        pass


class PyTorchTransformationSet(TransformationSet,Iterable[PyTorchTransformation]):

    def parameter_range(self)->[torch.Tensor]:
        parameters = torch.stack([t.parameters() for t in self],dim=0)
        mi,_ = parameters.min(dim=0)
        ma,_ = parameters.max(dim=0)
        return mi, ma