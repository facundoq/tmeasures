import abc
from typing import Iterable, List

import torch

from tmeasures import Transformation, TransformationSet


class PyTorchTransformation(Transformation,torch.nn.Module):

    @abc.abstractmethod
    def parameters(self)->torch.Tensor:
        pass


class PyTorchTransformationSet(TransformationSet,Iterable[PyTorchTransformation]):

    def parameter_range(self)->List[torch.Tensor]:
        parameters = torch.stack([t.parameters() for t in self],dim=0)
        mi,_ = parameters.min(dim=0)
        ma,_ = parameters.max(dim=0)
        return mi, ma


class IdentityTransformation(PyTorchTransformation):
    
    def parameters(self) -> torch.Tensor:
        return 0
     
    def __call__(self, x):
        return x


class IdentityTransformationSet(PyTorchTransformationSet):
    
    def __init__(self):
        super().__init__([IdentityTransformation()])