import tmeasures as tm
from torch.utils.data import Dataset
from .base import PyTorchMeasure, PyTorchMeasureOptions, PyTorchMeasureResult
from .activations_iterator import PytorchActivationsIterator
from . import ActivationsModule
from .layer_measures import Variance
from .quotient import QuotientMeasure
from .measure_transformer import MeasureTransformation, NoTransformation


class TransformationVarianceInvariance(PyTorchMeasure):

    def eval(self, dataset: Dataset, transformations: tm.TransformationSet, model: ActivationsModule,
             o: PyTorchMeasureOptions):
        dataset2d = tm.pytorch.dataset2d.SampleTransformationDataset(dataset, transformations, device=o.data_device)
        iterator = PytorchActivationsIterator(model, dataset2d, o)
        results = iterator.evaluate(Variance())
        return PyTorchMeasureResult(results, model.activation_names(), self)


class SampleVarianceInvariance(PyTorchMeasure):

    def eval(self, dataset: Dataset, transformations: tm.TransformationSet, model: ActivationsModule,
             o: PyTorchMeasureOptions):
        dataset2d = tm.pytorch.dataset2d.TransformationSampleDataset(dataset, transformations, device=o.data_device)
        iterator = PytorchActivationsIterator(model, dataset2d, o)
        results = iterator.evaluate(Variance())
        return PyTorchMeasureResult(results, model.activation_names(), self)


class NormalizedVarianceInvariance(QuotientMeasure):

    def __init__(self, measure_transformation: MeasureTransformation = NoTransformation()):
        super().__init__(TransformationVarianceInvariance(), SampleVarianceInvariance(),
                         measure_transformation=measure_transformation)

    def __repr__(self):
        return f"{self.abbreviation()}({self.measure_transformation})"
