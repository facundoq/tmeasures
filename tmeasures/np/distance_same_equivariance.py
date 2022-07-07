from typing import List
from .activations_iterator import ActivationsIterator
from tmeasures import MeasureResult
from . import NumpyMeasure,MeasureTransformation
from .stats_running import RunningMeanWelford
from .quotient import divide_activations

from .distance_invariance import DistanceAggregation

def list_get_all(list:List,indices:List[int])->List:
    return [list[i] for i in indices]

class BaseDistanceSameEquivarianceMeasure(NumpyMeasure):
    def __init__(self, distance_aggregation:DistanceAggregation):
        super().__init__()
        self.distance_aggregation=distance_aggregation

class TransformationDistanceSameEquivariance(BaseDistanceSameEquivarianceMeasure):
    def __repr__(self):
        return f"TDSE(da={self.distance_aggregation})"

    def name(self)->str:
        return "Transformation Distance Same-Equivariance"
    def abbreviation(self):
        return "TDSE"


    def eval(self,activations_iterator:ActivationsIterator,verbose=False)->MeasureResult:
        activations_iterator = activations_iterator.get_inverted_activations_iterator()
        mean_running= None

        for x, transformation_activations_iterator in activations_iterator.samples_first():
            # transformation_activations_iterator can iterate over all transforms
            for x_transformed, activations in transformation_activations_iterator:
                if mean_running is None:
                    mean_running = [RunningMeanWelford() for i in range(len(activations))]
                for j, layer_activations in enumerate(activations):
                    layer_measure= self.distance_aggregation.apply(layer_activations)
                    # update the mean over all transformation
                    mean_running[j].update(layer_measure)
        # calculate the final mean over all samples (and layers)
        means = [b.mean() for b in mean_running]
        return MeasureResult(means,activations_iterator.layer_names(),self)



class SampleDistanceSameEquivariance(BaseDistanceSameEquivarianceMeasure):

    def __repr__(self):
        return f"SDSE(da={self.distance_aggregation})"

    def name(self)->str:
        return "Sample Distance Same-Equivariance"
    def abbreviation(self):
        return "SDSE"


    def eval(self,activations_iterator:ActivationsIterator,verbose=False)->MeasureResult:
        activations_iterator = activations_iterator.get_inverted_activations_iterator()
        mean_running= None

        for transformation, samples_activations_iterator in activations_iterator.transformations_first():

            # transformation_activations_iterator can iterate over all transforms
            for x,activations in samples_activations_iterator:
                if mean_running is None:
                    mean_running = [RunningMeanWelford() for i in range(len(activations))]
                for j, layer_activations in enumerate(activations):
                    layer_measure= self.distance_aggregation.apply(layer_activations)
                    # update the mean over all transformation
                    mean_running[j].update(layer_measure)
        # calculate the final mean over all samples (and layers)
        means = [b.mean() for b in mean_running]
        return MeasureResult(means,activations_iterator.layer_names(),self)


class NormalizedDistanceSameEquivariance(NumpyMeasure):
    def __init__(self, distance_aggregation:DistanceAggregation):
        super().__init__()
        self.distance_aggregation=distance_aggregation
        self.transformation_measure=TransformationDistanceSameEquivariance(distance_aggregation)
        self.sample_measure=SampleDistanceSameEquivariance(distance_aggregation)


    def __repr__(self):
        return f"NDSE(da={self.distance_aggregation})"

    def name(self)->str:
        return "Normalized Distance Same-Equivariance"
    def abbreviation(self):
        return "NDSE"
    transformation_key=TransformationDistanceSameEquivariance.__name__
    sample_key=SampleDistanceSameEquivariance.__name__

    def eval(self,activations_iterator:ActivationsIterator,verbose=False) ->MeasureResult:

        transformation_result = self.transformation_measure.eval(activations_iterator,verbose)
        sample_result = self.sample_measure.eval(activations_iterator,verbose)
        result=divide_activations(transformation_result.layers,sample_result.layers)

        extra_values={ self.transformation_key:transformation_result,
                       self.sample_key:sample_result,
                       }
        return MeasureResult(result, transformation_result.layer_names,self,extra_values=extra_values)
