import tmeasures as tm


from tmeasures import MeasureResult
from .activations_iterator import ActivationsIterator
from .stats_running import RunningMeanAndVarianceWelford, RunningMeanWelford,RunningMeanVarianceSets
from .base import NumpyMeasure
from .quotient import divide_activations
from tqdm import tqdm
from .layer_transformation import MeasureTransformation,IdentityTransformation
import numpy as np

class TransformationVarianceInvariance(NumpyMeasure):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return f"{self.abbreviation()}"

    def eval(self, activations_iterator: ActivationsIterator,verbose=False) -> MeasureResult:
        layer_names = activations_iterator.layer_names()
        n_layers = len(layer_names)
        mean_running = [RunningMeanWelford() for i in range(n_layers)]

        for x,transformation_activations  in tqdm(activations_iterator.samples_first(),disable=not verbose):

            #calculate the running mean/variance/std over all transformations of x
            transformation_variances_running = [RunningMeanVarianceSets() for i in range(n_layers)]
            # print(transformation_variances_running[0])
            for x_transformed, activations in transformation_activations:
                for i, layer_activations in enumerate(activations):
                    # update the mean over all transformations for this sample
                    transformation_variances_running[i].update_batch(layer_activations.astype(np.double))

            # update the mean with the numpy sample of all transformations of x
            for i in range(n_layers):
                layer_measure = transformation_variances_running[i].std()

                # print(layer_measure)
                mean_running[i].update(layer_measure)

        # calculate the final mean over all samples (for each layer)
        mean_variances = [b.mean() for b in mean_running]
        return MeasureResult(mean_variances, layer_names, self)

class SampleVarianceInvariance(NumpyMeasure):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return f"{self.abbreviation()}"


    def eval(self, activations_iterator: ActivationsIterator,verbose=False) -> MeasureResult:
        layer_names = activations_iterator.layer_names()
        n_layers = len(layer_names)
        mean_variances_running = [RunningMeanWelford() for i in range(n_layers)]

        for transformation, samples_activations_iterator in tqdm(activations_iterator.transformations_first(),disable=not verbose):
            samples_variances_running = [RunningMeanAndVarianceWelford() for i in range(n_layers)]

            # calculate the variance of all samples for this transformation
            for x, batch_activations in samples_activations_iterator:
                for j, layer_activations in enumerate(batch_activations):
                    samples_variances_running[j].update_all(layer_activations)
            # update the mean over all transformation (and layers)
            for layer_mean_variances_running, layer_samples_variance_running in zip(mean_variances_running,samples_variances_running):
                samples_variance = layer_samples_variance_running.std()
                layer_mean_variances_running.update(samples_variance)

        # calculate the final mean over all transformations (and layers)

        mean_variances = [b.mean() for b in mean_variances_running]
        return MeasureResult(mean_variances, layer_names, self)


class NormalizedVarianceInvariance(NumpyMeasure):
    def __init__(self, pre_normalization_transformation: MeasureTransformation=IdentityTransformation()):
        self.sv = SampleVarianceInvariance()
        self.tv = TransformationVarianceInvariance()
        self.pre_normalization_transformation = pre_normalization_transformation

    def eval(self, activations_iterator: ActivationsIterator,verbose=False) -> MeasureResult:
        tv_result=self.tv.eval(activations_iterator,verbose)
        sv_result=self.sv.eval(activations_iterator,verbose)

        tv_result = self.pre_normalization_transformation.apply(tv_result)
        sv_result = self.pre_normalization_transformation.apply(sv_result)

        result=divide_activations(tv_result.layers, sv_result.layers)
        return MeasureResult(result, activations_iterator.layer_names(), self)

    def __repr__(self):
        ca = f"pnt={self.pre_normalization_transformation}"
        return f"{self.abbreviation()}({ca})"
