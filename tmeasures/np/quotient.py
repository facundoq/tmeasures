from .base import NumpyMeasure
from ..measure import ActivationsByLayer
from .. import MeasureResult
from .activations_iterator import ActivationsIterator
import numpy as np


def divide_activations(num:ActivationsByLayer, den:ActivationsByLayer)->ActivationsByLayer:
    #TODO evaluate other implementations
    # https://stackoverflow.com/questions/26248654/how-to-return-0-with-divide-by-zero
    eps = 0

    measures = []  # coefficient of variations
    for num_values,den_values in zip(num, den):
        # print(layer_v_transformations.shape,layer_v_samples.shape)

        normalized_measure = num_values.copy()

        normalized_measure[den_values  > eps] /= den_values [den_values  > eps]
        both_below_eps = np.logical_and(den_values  <= eps,
                                        num_values <= eps)
        normalized_measure[both_below_eps] = 1

        only_baseline_below_eps = np.logical_and(
            den_values  <= eps,
            num_values > eps)
        # print("num", np.where(num_values > eps))
        # print("den", np.where(den_values <= eps))
        # print("both", np.where(only_baseline_below_eps))

        normalized_measure[only_baseline_below_eps] = np.inf
        measures.append(normalized_measure)
    return measures

class QuotientMeasure(NumpyMeasure):
    def __init__(self, numerator_measure:NumpyMeasure, denominator_measure:NumpyMeasure):
        super().__init__()
        self.numerator_measure=numerator_measure
        self.denominator_measure=denominator_measure

    def __repr__(self):
        return f"QM({self.numerator_measure}_DIV_{self.denominator_measure})"

    def eval(self,activations_iterator:ActivationsIterator,verbose=False)->MeasureResult:
        v_transformations = self.numerator_measure.eval(activations_iterator,verbose=False)
        v_samples=self.denominator_measure.eval(activations_iterator,verbose=False)
        v=divide_activations(v_transformations.layers, v_samples.layers)

        layer_names = activations_iterator.layer_names()
        return MeasureResult(v,layer_names,self)


