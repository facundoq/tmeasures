from . import PyTorchMeasure, ActivationsByLayer, PyTorchMeasureOptions, ActivationsModule, PyTorchMeasureResult
from .. import MeasureResult, Measure
from torch.utils.data import Dataset
import torch
from .. import TransformationSet
from .measure_transformer import MeasureTransformation, NoTransformation
import tqdm.auto as tqdm


def safe_divide(x: torch.Tensor, y: torch.Tensor):
    eps = 0
    r = x.clone()

    r[y > eps] /= y[y > eps]
    both_below_eps = torch.logical_and(y <= eps,
                                       x <= eps)
    r[both_below_eps] = 1

    only_baseline_below_eps = torch.logical_and(y <= eps, x > eps)
    # print("num", np.where(num_values > eps))
    # print("den", np.where(den_values <= eps))
    # print("both", np.where(only_baseline_below_eps))

    r[only_baseline_below_eps] = float("Inf")
    return r


def divide_activations(num: ActivationsByLayer, den: ActivationsByLayer) -> ActivationsByLayer:
    # TODO evaluate other implementations
    # https://stackoverflow.com/questions/26248654/how-to-return-0-with-divide-by-zero
    return [safe_divide(x, y) for (x, y) in zip(num, den)]


class QuotientMeasureResult(PyTorchMeasureResult):
    def __init__(self, v: ActivationsByLayer, layer_names: [str], m: Measure, x_result: PyTorchMeasureResult,
                 y_result: PyTorchMeasureResult):
        super().__init__(v, layer_names, m)
        self.x_result = x_result
        self.y_result = y_result

    def numpy(self):
        self_numpy=super().numpy()
        self_numpy.x_result = self.x_result.numpy()
        self_numpy.y_result = self.y_result.numpy()
        return self_numpy


class QuotientMeasure(PyTorchMeasure):
    def __init__(self, numerator_measure: PyTorchMeasure, denominator_measure: PyTorchMeasure,
                 measure_transformation: MeasureTransformation = NoTransformation()):
        super().__init__()
        self.numerator_measure = numerator_measure
        self.denominator_measure = denominator_measure
        self.measure_transformation = measure_transformation

    def __repr__(self):
        return f"QM({self.numerator_measure}_DIV_{self.denominator_measure},{self.measure_transformation})"

    def eval(self, dataset: Dataset, transformations: TransformationSet, model: ActivationsModule,
             o: PyTorchMeasureOptions) -> QuotientMeasureResult:
        with tqdm.tqdm(total=2, disable=not o.verbose, colour='green') as pbar:
            v_transformations = self.numerator_measure.eval(dataset, transformations, model, o)
            self.measure_transformation.transform_result(v_transformations)
            pbar.update(1)
            v_samples = self.denominator_measure.eval(dataset, transformations, model, o)
            self.measure_transformation.transform_result(v_samples)
            pbar.update(2)
        v = [safe_divide(x, y) for (x, y) in zip(v_transformations.layers, v_samples.layers)]

        return QuotientMeasureResult(v, model.activation_names(), self, v_transformations, v_samples)
