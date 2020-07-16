from transformational_measures import ConvAggregation, MeasureResult
from transformational_measures import MeasureFunction
from transformational_measures.activations_iterator import ActivationsIterator
from transformational_measures.numpy.stats_running import RunningMeanAndVarianceWelford, RunningMeanWelford
from .base import NumpyMeasure
from .quotient import divide_activations

class TransformationVariance(NumpyMeasure):
    def __init__(self, measure_function:MeasureFunction=MeasureFunction.std):
        super().__init__()
        self.measure_function = measure_function

    def __repr__(self):
        if self.measure_function == MeasureFunction.std:
            mf=""
        else:
            mf =f"f={self.measure_function.value}"
        return f"TV({mf})"

    def eval(self, activations_iterator: ActivationsIterator) -> MeasureResult:
        layer_names = activations_iterator.layer_names()
        n_layers = len(layer_names)
        mean_running = [RunningMeanWelford() for i in range(n_layers)]
        for x,transformation_activations  in activations_iterator.samples_first():

            #calculate the running mean/variance/std over all transformations of x
            transformation_variances_running = [RunningMeanAndVarianceWelford() for i in range(n_layers)]
            for x_transformed, activations in transformation_activations:
                for i, layer_activations in enumerate(activations):
                    # apply function to conv layers
                    # update the mean over all transformations for this sample
                    transformation_variances_running[i].update_all(layer_activations)
            # update the mean with the numpy sample of all transformations of x
            for i in range(n_layers):
                layer_measure = self.measure_function.apply_running(transformation_variances_running[i])
                mean_running[i].update(layer_measure)

        # calculate the final mean over all samples (for each layer)
        mean_variances = [b.mean() for b in mean_running]
        return MeasureResult(mean_variances, layer_names, self)
    def name(self):
        return "Transformation Variance"
    def abbreviation(self):
        return "TV"

class SampleVariance(NumpyMeasure):
    def __init__(self, measure_function: MeasureFunction=MeasureFunction.std):
        super().__init__()
        self.measure_function = measure_function

    def __repr__(self):
        if self.measure_function == MeasureFunction.std:
            mf=""
        else:
            mf =f"f={self.measure_function.value}"
        return f"SV({mf})"


    def eval(self, activations_iterator: ActivationsIterator) -> MeasureResult:
        layer_names = activations_iterator.layer_names()
        n_layers = len(layer_names)
        mean_variances_running = [RunningMeanWelford() for i in range(n_layers)]

        for transformation, samples_activations_iterator in activations_iterator.transformations_first():
            samples_variances_running = [RunningMeanAndVarianceWelford() for i in range(n_layers)]
            # calculate the variance of all samples for this transformation
            for x, batch_activations in samples_activations_iterator:
                for j, layer_activations in enumerate(batch_activations):
                    samples_variances_running[j].update_all(layer_activations)
            # update the mean over all transformation (and layers)
            for layer_mean_variances_running, layer_samples_variance_running in zip(mean_variances_running,samples_variances_running):
                samples_variance = self.measure_function.apply_running(layer_samples_variance_running)
                layer_mean_variances_running.update(samples_variance)

        # calculate the final mean over all transformations (and layers)

        mean_variances = [b.mean() for b in mean_variances_running]
        return MeasureResult(mean_variances, layer_names, self)
    def name(self):
        return "Sample Variance"
    def abbreviation(self):
        return "SV"



class NormalizedVariance(NumpyMeasure):
    def __init__(self, conv_aggregation: ConvAggregation,measure_function: MeasureFunction=MeasureFunction.std):
        self.sv = SampleVariance(measure_function)
        self.tv = TransformationVariance(measure_function)
        self.measure_function = measure_function
        self.conv_aggregation = conv_aggregation

    def eval(self, activations_iterator: ActivationsIterator) -> MeasureResult:
        tv_result=self.tv.eval(activations_iterator)
        sv_result=self.sv.eval(activations_iterator)

        tv_result = tv_result.collapse_convolutions(self.conv_aggregation)
        sv_result = sv_result.collapse_convolutions(self.conv_aggregation)

        result=divide_activations(tv_result.layers,sv_result.layers)
        return MeasureResult(result, activations_iterator.layer_names(), self)

    def __repr__(self):
        if self.measure_function == MeasureFunction.std:
            mf = ""
        else:
            mf = f"f={self.measure_function.value}"

        if self.conv_aggregation == ConvAggregation.none:
            ca =""
        else:
            ca = f"ca={self.conv_aggregation.value}"
        if ca!="" and mf!="":
            sep = ","
        else:
                sep = ""
        return f"NV({ca}{sep}{mf})"
    def name(self):
        return "Normalized Variance"
    def abbreviation(self):
        return "NV"