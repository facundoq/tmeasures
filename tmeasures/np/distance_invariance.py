from . import MeasureTransformation, NumpyMeasure, IdentityTransformation
from .activations_iterator import ActivationsIterator
from tmeasures import MeasureResult
from .stats_running import RunningMeanWelford
from .quotient import divide_activations
from .aggregation import DistanceAggregation
import tmeasures as tm


class TransformationDistanceInvariance(NumpyMeasure):
    def __init__(self, distance_aggregation: DistanceAggregation):
        super().__init__()
        self.distance_aggregation = distance_aggregation

    def __repr__(self):
        return f"{self.abbreviation()}(da={self.distance_aggregation})"

    def eval(self, activations_iterator: ActivationsIterator, verbose=False) -> MeasureResult:
        layer_names = activations_iterator.layer_names()
        n_intermediates = len(layer_names)
        mean_running = [RunningMeanWelford() for i in range(n_intermediates)]
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
        return MeasureResult(mean_variances, layer_names, self)

    def name(self):
        return "Transformation Distance"

    def abbreviation(self):
        return "TD"


class SampleDistanceInvariance(NumpyMeasure):
    def __init__(self, distance_aggregation: DistanceAggregation):
        super().__init__()
        self.distance_aggregation = distance_aggregation

    def __repr__(self):
        return f"{self.abbreviation()}(da={self.distance_aggregation})"

    def eval(self, activations_iterator: ActivationsIterator, verbose=False) -> MeasureResult:
        layer_names = activations_iterator.layer_names()
        n_layers = len(layer_names)
        mean_running = [RunningMeanWelford() for i in range(n_layers)]

        for transformation, transformation_activations in activations_iterator.transformations_first():
            # calculate the variance of all samples for this transformation
            for x, batch_activations in transformation_activations:
                for j, layer_activations in enumerate(batch_activations):
                    layer_measure = self.distance_aggregation.apply(layer_activations)
                    mean_running[j].update(layer_measure)

        # calculate the final mean over all transformations (and layers)
        mean_variances = [b.mean() for b in mean_running]
        return MeasureResult(mean_variances, layer_names, self)


class NormalizedDistanceInvariance(NumpyMeasure):
    def __init__(self, distance_aggregation: DistanceAggregation,
                 pre_normalization_transformation: MeasureTransformation = IdentityTransformation()):
        self.distance_aggregation = distance_aggregation
        self.td = TransformationDistanceInvariance(distance_aggregation)
        self.sd = SampleDistanceInvariance(distance_aggregation)
        self.pre_normalization_transformation = pre_normalization_transformation

    def eval(self, activations_iterator: ActivationsIterator, verbose=False) -> MeasureResult:
        td_result = self.td.eval(activations_iterator, verbose)
        sd_result = self.sd.eval(activations_iterator, verbose)

        td_result = self.pre_normalization_transformation.apply(td_result)
        sd_result = self.pre_normalization_transformation.apply(sd_result)

        result = divide_activations(td_result.layers, sd_result.layers)
        return MeasureResult(result, activations_iterator.layer_names(), self)

    def __repr__(self):
        return f"{self.abbreviation()}(pnt={self.pre_normalization_transformation},da={self.distance_aggregation})"
