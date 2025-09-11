from torch.utils.data import Dataset

from .activations_iterator import PytorchActivationsIterator
from .base import PyTorchLayerMeasure, PyTorchMeasure, PyTorchMeasureOptions, PyTorchMeasureResult
from .dataset2d import TransformationSampleDataset
from .model import ActivationsModule
from .transformations import PyTorchTransformationSet


class PyTorchMeasureByLayer(PyTorchMeasure):

    def __init__(self,layer_measure:PyTorchLayerMeasure) -> None:
        super().__init__()
        self.layer_measure=layer_measure

    def eval(self, dataset: Dataset, transformations: PyTorchTransformationSet
    , model: ActivationsModule,
             o: PyTorchMeasureOptions) -> PyTorchMeasureResult:
        dataset2d = TransformationSampleDataset(dataset, transformations, device=o.data_device)
        iterator = PytorchActivationsIterator(model, dataset2d, o)
        results = iterator.evaluate(self.layer_measure)
        return PyTorchMeasureResult(results, model.activation_names(), self)
