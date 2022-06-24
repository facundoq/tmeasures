import tmeasures as tm
from torch.utils.data import Dataset
from .base import PyTorchMeasure, PyTorchMeasureOptions, PyTorchMeasureResult
from .activations_iterator import PytorchActivationsIterator
from . import ActivationsModule, Variance
from .. import InvertibleTransformation
from .quotient import QuotientMeasure
import torch

from .activations_iterator import ActivationsTransformer
from .measure_transformer import MeasureTransformation, NoTransformation


class InverseTransformationTransformer(ActivationsTransformer):

    def transform(self, activations: torch.Tensor, x: torch.Tensor,
                  transformations: [InvertibleTransformation]) -> torch.Tensor:
        transformed = []
        for i in range(activations.shape[0]):
            inverse = transformations[i].inverse()
            transformed.append(inverse(activations[i,]))
        return torch.stack(transformed, dim=0)


class TransformationVarianceSameEquivariance(PyTorchMeasure):

    def eval(self, dataset: Dataset, transformations: tm.TransformationSet, model: ActivationsModule,
             o: PyTorchMeasureOptions):
        dataset2d = tm.pytorch.dataset2d.SampleTransformationDataset(dataset, transformations, device=o.data_device)
        iterator = PytorchActivationsIterator(model, dataset2d, o,
                                              activations_transformer=InverseTransformationTransformer())
        results = iterator.evaluate(Variance())
        return PyTorchMeasureResult(results, model.activation_names(), self)


class SampleVarianceSameEquivariance(PyTorchMeasure):

    def eval(self, dataset: Dataset, transformations: tm.TransformationSet, model: ActivationsModule,
             o: PyTorchMeasureOptions):
        dataset2d = tm.pytorch.dataset2d.TransformationSampleDataset(dataset, transformations, device=o.data_device)
        iterator = PytorchActivationsIterator(model, dataset2d, o,
                                              activations_transformer=InverseTransformationTransformer())
        results = iterator.evaluate(Variance())
        return PyTorchMeasureResult(results, model.activation_names(), self)


class NormalizedVarianceSameEquivariance(QuotientMeasure):

    def __init__(self, measure_transformation: MeasureTransformation = NoTransformation()):
        super().__init__(TransformationVarianceSameEquivariance(), SampleVarianceSameEquivariance(),
                         measure_transformation=measure_transformation)

    def __repr__(self):
        extra =""
        return f"{self.abbreviation()}({self.measure_transformation})"
