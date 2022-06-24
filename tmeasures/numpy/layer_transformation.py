import numpy as np
from abc import ABC,abstractmethod
from enum import Enum
from .. import MeasureResult
from typing import Tuple

class MeasureTransformation(ABC):

    @abstractmethod
    def apply(self,r:MeasureResult):
        pass
    def __repr__(self):
        return self.__class__.__name__

class IdentityTransformation(MeasureTransformation):
    def apply(self,r:MeasureResult):
        return r


class AggregateFunction(Enum):
    mean = "mean"
    max = "max"
    sum = "sum"

    def functions(self):
        return {AggregateFunction.mean: np.nanmean
            , AggregateFunction.sum: np.nansum
            , AggregateFunction.max: np.nanmax
                }
    def get_function(self):
        return self.functions()[self]
    def __repr__(self):
        return self.value


class AggregateTransformation(MeasureTransformation):

    def __init__(self, f:AggregateFunction=AggregateFunction.mean, axis:Tuple[int]=None):
         self.f = f
         if axis is None:
            self.axis=(0,) # keep only axis 0
         else:
            assert len(axis) > 0
            self.axis = axis # axis to keep

    def apply(self, r:MeasureResult)->MeasureResult:
        aggregate_function= self.f.get_function()
        results=[]
        for layer in r.layers:
            # only apply to layers with at least max(self.axis) dims
            dims= len(layer.shape)
            if max(self.axis) < dims:
                diff = set(range(dims)).difference(set(self.axis))
                diff = tuple(diff)
                flat_activations = aggregate_function(layer,axis=diff)
                # assert len(flat_activations.shape) == len(self.axis),f"After collapsing, the activation shape should have only {len(self.axis)} dimensions. Found vector with shape {flat_activations.shape} instead."
            else:
                flat_activations = layer.copy()
            results.append(flat_activations)

        return MeasureResult(results,r.layer_names,r.measure,r.extra_values)
    def __repr__(self):
        return f"{self.__class__.__name__}(f={self.f.value},axis={self.axis})"




# def apply(self, layer:np.ndarray) -> np.ndarray:
#     '''
#     :param layer:  a 4D np array (else apply no aggregation)
#     :return:
#     '''
#
#     # none does no aggregation
#     if self == ConvAggregation.none:
#         return layer
#     layer=np.abs(layer)
#     # only aggregate conv layers (n,c,h,w)
#     if len(layer.shape) != 4:
#         return layer
#
#     function = self.functions()[self]
#     n, c, h, w = layer.shape
#     flat_activations = np.zeros((n, c))
#     for i in range(n):
#         flat_activations[i, :] = function(layer[i, :, :, :], axis=(1,2))
#
#     return flat_activations
#

class AggregateConvolutions(AggregateTransformation):
    def __init__(self, f:AggregateFunction=AggregateFunction.mean):
        super().__init__(f,(0,))
