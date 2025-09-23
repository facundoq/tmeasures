import abc
import typing
from typing import List

import torch
from torch.utils.data import Dataset

import tmeasures as tm

from .. import MeasureResult, StratifiedMeasureResult
from . import ActivationsModule

ActivationsByLayer = [torch.Tensor]

CPU = torch.device("cpu")
class PyTorchMeasureOptions:
    def __init__(self, batch_size=32, num_workers=0, verbose=True, model_device=CPU, measure_device=CPU,
                 data_device=CPU):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.verbose = verbose
        self.model_device = model_device
        self.measure_device = measure_device
        self.data_device = data_device
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(b={self.batch_size},workers={self.num_workers},devices[model,measure,data]=[{self.model_device,self.measure_device,self.data_device}])"


STRowIterator = typing.Union[typing.Iterable[torch.Tensor], typing.Sized]
STMatrixIterator = typing.Union[typing.Iterable[STRowIterator], typing.Sized]


class PyTorchActivationMeasure:

    @abc.abstractmethod
    def eval(self, iterator: STMatrixIterator,layer_name:str) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def generate_result(self, layer_results: ActivationsByLayer, layer_names: List[str]):
        pass


class PyTorchMeasureResult(tm.MeasureResult):

    def numpy(self):
        self.layers = [l.cpu().numpy() for l in self.layers]
        return self


class PyTorchMeasure(tm.Measure):
    def __repr__(self):
        return f"{self.abbreviation()}"

    @abc.abstractmethod
    def eval(self, dataset: Dataset, transformations: tm.TransformationSet, model: ActivationsModule,
             o: PyTorchMeasureOptions) -> PyTorchMeasureResult:
        '''

        '''
        pass

    def eval_stratified(self, datasets: typing.List[Dataset], transformations: tm.TransformationSet,
                        model: ActivationsModule, o: PyTorchMeasureOptions,
                        labels: typing.List[str]) -> StratifiedMeasureResult:
        '''
        Calculate the `variance_measure` for each class separately
        Also calculate the average stratified `variance_measure` over all classes
        '''
        results_per_set = [self.eval(dataset,transformations,model,o) for dataset in datasets]
        stratified_measure_layers = self.mean_result(results_per_set)
        stratified_measure_layers = [l.cpu().numpy() for l in stratified_measure_layers]

        layer_names = results_per_set[0].layer_names

        return StratifiedMeasureResult(stratified_measure_layers, layer_names, self, results_per_set, labels)

    def mean_result(self, measure_results: typing.List[MeasureResult]) -> ActivationsByLayer:
        # calculate the mean activation of each unit in each layer over the datasets
        results = [r.layers for r in measure_results]
        # results is a list (classes) of list layers)
        # turn it into a list (layers) of lists (classes)
        layer_class_vars = [list(i) for i in zip(*results)]
        # compute average result of each layer over datasets
        layer_vars = [sum(layer_values) / len(layer_values) for layer_values in layer_class_vars]
        return layer_vars


