import torch
import transformational_measures as tm
from torch.utils.data import Dataset
from . import ObservableLayersModule
from .. import MeasureResult
import abc
import typing
ActivationsByLayer = [torch.Tensor]

class PyTorchMeasureOptions:
    def __init__(self, batch_size=32, num_workers=0, verbose=True,model_device="cpu",measure_device="cpu"):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.verbose = verbose
        self.model_device = model_device
        self.measure_device = measure_device

STRowIterator = typing.Union[typing.Iterable[torch.Tensor],typing.Sized]
STMatrixIterator = typing.Union[typing.Iterable[STRowIterator],typing.Sized]

class PyTorchLayerMeasure:

    @abc.abstractmethod
    def eval(self,iterator:STMatrixIterator)->torch.Tensor:
        pass

    @abc.abstractmethod
    def generate_result(self,layer_results:ActivationsByLayer,layer_names:[str]):
        pass

class PyTorchMeasureResult(tm.MeasureResult):

    def numpy(self):
        for l in self.layers:
            l[:]=l.numpy()
        return self

class PyTorchMeasure(tm.Measure):
    def __repr__(self):
        return f"{self.abbreviation()}"

    @abc.abstractmethod
    def eval(self,dataset:Dataset,transformations:tm.TransformationSet,model:ObservableLayersModule,o:PyTorchMeasureOptions)-> MeasureResult:
        '''

        '''
        pass
