from torch.utils.data import Dataset

from .. import BaseActivationsModule
from ..activations_iterator import PytorchActivationsIterator
from ..base import PyTorchMeasure, PyTorchMeasureOptions, PyTorchMeasureResult
from ..dataset2d import SampleTransformationDataset, TransformationSampleDataset
from ..layer_measures import Variance
from ..measure_transformer import MeasureTransformation, NoTransformation
from .quotient import QuotientMeasure
from ..transformations import PyTorchTransformationSet


class TransformationVarianceInvariance(PyTorchMeasure):

    def eval(self, dataset: Dataset, transformations: PyTorchTransformationSet, model: BaseActivationsModule,
             o: PyTorchMeasureOptions):
        dataset2d = SampleTransformationDataset(dataset, transformations, device=o.data_device)
        iterator = PytorchActivationsIterator(model, dataset2d, o)
        results = iterator.evaluate(Variance())
        return PyTorchMeasureResult(results, model.activation_names(), self)


class SampleVarianceInvariance(PyTorchMeasure):

    def eval(self, dataset: Dataset,  transformations:PyTorchTransformationSet, model: BaseActivationsModule,
             o: PyTorchMeasureOptions):
        dataset2d = TransformationSampleDataset(dataset, transformations, device=o.data_device)
        iterator = PytorchActivationsIterator(model, dataset2d, o)
        results = iterator.evaluate(Variance())
        return PyTorchMeasureResult(results, model.activation_names(), self)


class NormalizedVarianceInvariance(QuotientMeasure):

    def __init__(self, measure_transformation: MeasureTransformation = NoTransformation()):
        super().__init__(TransformationVarianceInvariance(), SampleVarianceInvariance(),
                         measure_transformation=measure_transformation)

    def __repr__(self):
        return f"{self.abbreviation()}({self.measure_transformation})"
