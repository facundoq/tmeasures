from __future__ import annotations
import abc
from typing import Iterable, List

import torch

from tmeasures import Transformation, TransformationSet


class PyTorchTransformation(torch.nn.Module,abc.ABC):

    @abc.abstractmethod
    def parameters(self)->torch.Tensor:
        pass

    @abc.abstractmethod
    def __call__(self, x:torch.Tensor)->torch.Tensor:
        pass

class PyTorchInvertibleTransformation(PyTorchTransformation):
    @abc.abstractmethod
    def inverse(self)->PyTorchInvertibleTransformation:
        pass

class PyTorchTransformationSet(TransformationSet,Iterable[PyTorchTransformation]):

    def parameter_range(self)->tuple[torch.Tensor,torch.Tensor]:
        parameters = torch.stack([t.parameters() for t in self],dim=0)
        mi,_ = parameters.min(dim=0)
        ma,_ = parameters.max(dim=0)
        return (mi, ma)

class PyTorchInvertibleTransformationSet(PyTorchTransformationSet,Iterable[PyTorchInvertibleTransformation]):
    pass



class IdentityTransformation(PyTorchInvertibleTransformation):

    def parameters(self) -> torch.Tensor:
        return torch.Tensor(0)
    def inverse(self)->IdentityTransformation:
        return self
    def __call__(self, x):
        return x


class IdentityTransformationSet(PyTorchTransformationSet):

    def __init__(self):
        super().__init__([IdentityTransformation()])
    def valid_input(self):
        return True
    def copy(self):
        return self
    def id(self):
        return "Identity"

