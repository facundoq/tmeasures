from .base import Measure,MeasureResult
from transformation_measure.iterators.activations_iterator import ActivationsIterator
from transformation_measure import ConvAggregation
import numpy as np
from transformation_measure.measure.stats_running import RunningMeanAndVarianceWelford,RunningMeanWelford
from typing import List
from enum import Enum
from .quotient import divide_activations
from sklearn.metrics.pairwise import euclidean_distances
import sklearn
from .aggregation import DistanceAggregation

class TransformationDistance(Measure):
    def __init__(self, distance_aggregation:DistanceAggregation):
        super().__init__()
        self.distance_aggregation=distance_aggregation

    def __repr__(self):
        return f"TD(da={self.distance_aggregation.name})"


    def eval(self,activations_iterator:ActivationsIterator)->MeasureResult:
        layer_names=activations_iterator.layer_names()
        n_intermediates = len(layer_names)
        mean_running= [RunningMeanWelford() for i in range(n_intermediates)]
        for x, transformation_activations_iterator in activations_iterator.samples_first():
            # transformation_activations_iterator can iterate over all transforms
            for x_transformed, activations in transformation_activations_iterator:
                for j, layer_activations in enumerate(activations):
                    # calculate the distance aggregation only for this batch
                    layer_measure = self.distance_aggregation.apply(layer_activations)
                    # update the mean over all transformation
                    mean_running[j].update(layer_measure)

        # calculate the final mean over all samples (and layers)
        mean_variances = [b.mean() for b in mean_running]
        return MeasureResult(mean_variances,layer_names,self)

    def name(self):
        return "Transformation Distance"
    def abbreviation(self):
        return "TD"


class SampleDistance(Measure):
    def __init__(self, distance_aggregation: DistanceAggregation):
        super().__init__()
        self.distance_aggregation = distance_aggregation

    def __repr__(self):
        return f"SD(da={self.distance_aggregation.name})"

    def eval(self,activations_iterator:ActivationsIterator)->MeasureResult:
        layer_names = activations_iterator.layer_names()
        n_layers = len(layer_names)
        mean_running = [RunningMeanWelford() for i in range(n_layers)]

        for transformation, transformation_activations in activations_iterator.transformations_first():
            # calculate the variance of all samples for this transformation
            for x, batch_activations in transformation_activations:
                for j, layer_activations in enumerate(batch_activations):
                    layer_measure = self.distance_aggregation.apply(layer_activations)
                    mean_running [j].update(layer_measure)

        # calculate the final mean over all transformations (and layers)
        mean_variances = [b.mean() for b in mean_running]
        return MeasureResult(mean_variances,layer_names,self)
    def name(self):
        return "Sample Distance"
    def abbreviation(self):
        return "SD"




class NormalizedDistance(Measure):
    def __init__(self, distance_aggregation: DistanceAggregation,conv_aggregation:ConvAggregation):
        self.distance_aggregation = distance_aggregation
        self.td = TransformationDistance(distance_aggregation)
        self.sd = SampleDistance(distance_aggregation)
        self.conv_aggregation=conv_aggregation

    def eval(self, activations_iterator: ActivationsIterator) -> MeasureResult:
        if self.distance_aggregation.keep_feature_maps and self.conv_aggregation != ConvAggregation.none:
            print("Warning: ConvAggregation strategies dot not have any effect when keep_feature_maps is True.")

        td_result = self.td.eval(activations_iterator)
        sd_result = self.sd.eval(activations_iterator)

        td_result = td_result.collapse_convolutions(self.conv_aggregation)
        sd_result = sd_result.collapse_convolutions(self.conv_aggregation)

        result = divide_activations(td_result.layers, sd_result.layers)
        return MeasureResult(result, activations_iterator.layer_names(), self)

    def __repr__(self):
        return f"ND(ca={self.conv_aggregation.value},da={self.distance_aggregation.name})"


    def name(self):
        return "Normalized Distance"
    def abbreviation(self):
        return "ND"