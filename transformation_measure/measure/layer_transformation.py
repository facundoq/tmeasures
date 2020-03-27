import numpy as np
from enum import Enum

class ConvAggregation(Enum):
    mean = "mean"
    max = "max"
    sum = "sum"
    none = "none"

    def functions(self):
        return {ConvAggregation.mean: np.nanmean
        , ConvAggregation.sum: np.nansum
        , ConvAggregation.max: np.nanmax
        }

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

    def apply3D(self,layer:np.ndarray,) -> np.ndarray:
        '''

        :param layer:  a 3D np array (else apply no aggregation)
        :return:
        '''
        if self == ConvAggregation.none:
            return layer
        # only aggregate conv layers (c,h,w)
        if len(layer.shape) != 3:
            return layer

        function = self.functions()[self]
        flat_activations = function(layer,axis=(1,2))

        return flat_activations