import transformational_measures as tm
from torch.utils.data import Dataset
from .base import PyTorchMeasure,PyTorchMeasureOptions, PyTorchMeasureResult
from .activations_iterator import PytorchActivationsIterator
from . import ObservableLayersModule
from .layer_measures import Variance
from .quotient import QuotientMeasure


class TransformationVarianceInvariance(PyTorchMeasure):

    def eval(self,dataset:Dataset,transformations:tm.TransformationSet,model:ObservableLayersModule,o:PyTorchMeasureOptions):
        dataset2d = tm.pytorch.dataset2d.SampleTransformationDataset(dataset, transformations)
        iterator = PytorchActivationsIterator(model, dataset2d,o)
        results = iterator.evaluate(Variance())
        return PyTorchMeasureResult(results,model.activation_names(),self)


class SampleVarianceInvariance(PyTorchMeasure):

    def eval(self,dataset:Dataset,transformations:tm.TransformationSet,model:ObservableLayersModule,o:PyTorchMeasureOptions):
        dataset2d = tm.pytorch.dataset2d.TransformationSampleDataset(dataset, transformations)
        iterator = PytorchActivationsIterator(model, dataset2d, o)
        results = iterator.evaluate(Variance())
        return PyTorchMeasureResult(results, model.activation_names(), self)

class NormalizedVarianceInvariance(QuotientMeasure):

    def __init__(self):
        super().__init__(TransformationVarianceInvariance(),SampleVarianceInvariance())
    def __repr__(self):
        return f"{self.abbreviation()}"
