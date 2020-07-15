from .base import NumpyMeasure
from transformation_measure.activations_iterator import ActivationsIterator
from transformation_measure import ConvAggregation, MeasureResult
from transformation_measure.numpy.stats_running import RunningMeanWelford
from .quotient import divide_activations
from .aggregation import DistanceAggregation


class TransformationDistance(NumpyMeasure):
    def __init__(self, distance_aggregation:DistanceAggregation):
        super().__init__()
        self.distance_aggregation=distance_aggregation

    def __repr__(self):
        return f"TD(da={self.distance_aggregation})"


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


class SampleDistance(NumpyMeasure):
    def __init__(self, distance_aggregation: DistanceAggregation):
        super().__init__()
        self.distance_aggregation = distance_aggregation

    def __repr__(self):
        return f"SD(da={self.distance_aggregation})"

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




class NormalizedDistance(NumpyMeasure):
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

        td_result = self.conv_aggregation.collapse_convolutions(td_result)
        sd_result = self.conv_aggregation.collapse_convolutions(sd_result)

        result = divide_activations(td_result.layers, sd_result.layers)
        return MeasureResult(result, activations_iterator.layer_names(), self)

    def __repr__(self):
        return f"ND(ca={self.conv_aggregation.value},da={self.distance_aggregation})"


    def name(self):
        return "Normalized Distance"
    def abbreviation(self):
        return "ND"