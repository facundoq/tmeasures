from ... import TransformationSet,Transformation
import abc
import torch
from typing import Iterable


class PyTorchTransformation(Transformation):

    @abc.abstractmethod
    def parameters(self)->torch.Tensor:
        pass

class PyTorchTransformationSet(TransformationSet,Iterable[PyTorchTransformation]):
    
    def parameter_range(self)->[torch.Tensor]:
        parameters = torch.stack([t.parameters() for t in self],dim=0)
        mi,_ = parameters.min(dim=0)
        ma,_ = parameters.max(dim=0)
        return mi, ma
    