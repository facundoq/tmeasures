from tmeasures import MeasureResult
from ..base import NumpyMeasure
from ..activations_iterator import ActivationsIterator
import numpy as np
from ..stats_running import RunningMeanWelford

from abc import abstractmethod

from typing import Callable
class EventDrivenLayerMeasure():

    def __init__(self,layer_index:int,layer_name:str):
        self.layer_name=layer_name
        self.layer_index=layer_index
    @abstractmethod
    def update_layer(self, activations:np.ndarray):
        pass

    @abstractmethod
    def get_final_result(self)->np.ndarray:
        pass
    def on_begin_iteration(self):
        pass
    def on_begin(self):
        pass

class LayerTransformationMeasure(NumpyMeasure):
    def __init__(self, layer_measure_generator:Callable[[int,str], EventDrivenLayerMeasure]):
        super().__init__()
        self.layer_measure_generator=layer_measure_generator

    def eval(self,activations_iterator:ActivationsIterator)->MeasureResult:
        layer_names=activations_iterator.layer_names()
        n_intermediates = len(layer_names)
        layer_measures = [self.layer_measure_generator(i,n) for i,n in enumerate(layer_names)]

        for r in layer_measures:
            r.on_begin()

        for activations, x_transformed in activations_iterator.samples_first():
            for r in layer_measures:
                r.on_begin_iteration()
            # activations has the activations for all the transformations
            for j, layer_activations in enumerate(activations):
                layer_measures[j].update_layer(layer_activations)

        results = [r.get_final_result() for r in layer_measures]
        return MeasureResult(results,layer_names,self)

#
# class LayerSampleMeasure(Measure):
#     def __init__(self, layer_measure_generator:Callable[[int,str], EventDrivenLayerMeasure]):
#         super().__init__()
#         self.layer_measure_generator=layer_measure_generator
#
#     def __repr__(self):
#         return f"(f={self.measure_function.value},da={self.distance_aggregation.value})"
#
#     def eval(self,activations_iterator:ActivationsIterator)->MeasureResult:
#         layer_names = activations_iterator.activation_names()
#         n_layers = len(layer_names)
#         mean_variances_running = [RunningMean() for i in range(n_layers)]
#
#         for transformation, transformation_activations in activations_iterator.transformations_first():
#             # calculate the variance of all samples for this transformation
#             for x, batch_activations in transformation_activations:
#                 for j, layer_activations in enumerate(batch_activations):
#                     for i in range(layer_activations.shape[0]):
#                         layer_measure = self.measure_function.apply(layer_activations)
#                         mean_variances_running [j].update(layer_measure )
#
#         # calculate the final mean over all transformations (and layers)
#         mean_variances = [b.mean() for b in mean_variances_running]
#         return MeasureResult(mean_variances,layer_names,self)


class STDLayerMeasure(EventDrivenLayerMeasure):

    def __init__(self,layer_index:int,layer_name:str):
        super(STDLayerMeasure).__init__(layer_index,layer_name)
        self.running_mean = RunningMeanWelford()

    def update_layer(self, activations):
        layer_measure = activations.std(axis=0)
        self.running_mean.update(layer_measure)

    def get_final_result(self):
        return self.running_mean.mean()
