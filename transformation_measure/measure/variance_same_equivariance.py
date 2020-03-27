from transformation_measure import ConvAggregation
from transformation_measure import MeasureFunction, QuotientMeasure
from transformation_measure.iterators.activations_iterator import ActivationsIterator
from transformation_measure.measure.stats_running import RunningMeanAndVarianceWelford, RunningMeanWelford,RunningMeanAndVarianceNaive
from .base import Measure, MeasureResult
from .quotient import divide_activations

class TransformationVarianceSameEquivariance(Measure):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return f"TVSE()"

    def eval(self, activations_iterator: ActivationsIterator) -> MeasureResult:
        activations_iterator = activations_iterator.get_inverted_activations_iterator()

        mean_running = None
        for x,transformation_activations  in activations_iterator.samples_first():
            transformation_variances_running = None
            #calculate the running mean/variance/std over all transformations of x
            for x_transformed, activations in transformation_activations:
                if mean_running is None:
                    n_layers = len(activations)
                    mean_running = [RunningMeanWelford() for i in range(n_layers)]
                if transformation_variances_running is None:
                    n_layers = len(activations)
                    transformation_variances_running = [RunningMeanAndVarianceWelford() for i in range(n_layers)]
                for i, layer_activations in enumerate(activations):
                    # apply function to conv layers
                    # update the mean over all transformations for this sample
                    transformation_variances_running[i].update_all(layer_activations)
            # update the mean with the measure sample of all transformations of x
            for i,layer_variance in enumerate(transformation_variances_running):
                mean_running[i].update(layer_variance.std())

        # calculate the final mean over all samples (for each layer)
        mean_variances = [b.mean() for b in mean_running]
        return MeasureResult(mean_variances, activations_iterator.layer_names(), self)
    def name(self):
        return "Transformation Variance Same Equivariance"
    def abbreviation(self):
        return "TVSE"

class SampleVarianceSameEquivariance(Measure):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return f"SVSE()"


    def eval(self, activations_iterator: ActivationsIterator) -> MeasureResult:
        activations_iterator:ActivationsIterator = activations_iterator.get_inverted_activations_iterator()
        ts = list(map(str, (activations_iterator.get_transformations())))
        print(f"{len(ts)} transformation")
        print("\n".join(ts))

        mean_variances_running = None

        for transformation, samples_activations_iterator in activations_iterator.transformations_first():
            samples_variances_running = None
            # calculate the variance of all samples for this transformation
            for x, batch_activations in samples_activations_iterator:
                if mean_variances_running is None:
                    n_layers = len(batch_activations)
                    mean_variances_running = [RunningMeanWelford() for i in range(n_layers)]
                if samples_variances_running is None:
                    n_layers = len(batch_activations)
                    samples_variances_running = [RunningMeanAndVarianceWelford() for i in range(n_layers)]
                for j, layer_activations in enumerate(batch_activations):
                    samples_variances_running[j].update_all(layer_activations)
            # update the mean over all transformation (and layers)
            for layer_mean_variances_running, layer_samples_variance_running in zip(mean_variances_running,samples_variances_running):
                layer_mean_variances_running.update(layer_samples_variance_running.std())

        # calculate the final mean over all transformations (and layers)

        mean_variances = [b.mean() for b in mean_variances_running]
        return MeasureResult(mean_variances, activations_iterator.layer_names(), self)
    def name(self):
        return "Sample Variance Same Equivariance"
    def abbreviation(self):
        return "SVSE"



class NormalizedVarianceSameEquivariance(Measure):
    def __init__(self, conv_aggregation: ConvAggregation):
        self.sv = SampleVarianceSameEquivariance()
        self.tv = TransformationVarianceSameEquivariance()
        self.conv_aggregation = conv_aggregation

    transformation_key=TransformationVarianceSameEquivariance.__name__
    sample_key=SampleVarianceSameEquivariance.__name__

    def eval(self, activations_iterator: ActivationsIterator) -> MeasureResult:

        sample_result=self.sv.eval(activations_iterator)
        transformation_result = self.tv.eval(activations_iterator)

        transformation_result = transformation_result.collapse_convolutions(self.conv_aggregation)
        sample_result = sample_result.collapse_convolutions(self.conv_aggregation)

        extra_values={ self.transformation_key:transformation_result,
                       self.sample_key:sample_result,
                       }

        result=divide_activations(transformation_result.layers,sample_result.layers)
        return MeasureResult(result, transformation_result.layer_names, self,extra_values)

    def __repr__(self):
        ca = f"ca={self.conv_aggregation.value}"

        return f"NVSE({ca})"
    def name(self):
        return "Normalized Variance Same Equivariance"
    def abbreviation(self):
        return "NVSE"