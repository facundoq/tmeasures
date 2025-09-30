import concurrent.futures
import typing
from collections.abc import Generator

import torch
import tqdm.auto as tqdm
from torch.utils.data import DataLoader

from tmeasures.pytorch.model import ActivationValues

# from .activations_transformer import ActivationsTransformer
from .. import InvertibleTransformation, Transformation
from . import BaseActivationsModule
from .base import PyTorchActivationMeasure, PyTorchMeasure, PyTorchMeasureOptions
from .computation_model import ThreadsComputationModel
from .dataset2d import Dataset2D, STDataset
from .transformations import PyTorchTransformation

try:
    import namedthreads
    namedthreads.patch()
except ImportError:
    pass

import abc
import threading
from typing import Callable, List

from .. import logger as tm_logger

logger = tm_logger.getChild("pytorch.activations_iterator")

class ActivationsTransformer(abc.ABC):

    @abc.abstractmethod
    def transform(self, activations: torch.Tensor, x: torch.Tensor, transformations: List[PyTorchTransformation]) -> torch.Tensor:
        pass


class IdentityActivationsTransformer(ActivationsTransformer):
    def transform(self, activations: torch.Tensor, x: torch.Tensor, transformations: List[PyTorchTransformation]) -> torch.Tensor:
        return activations





class PytorchActivationsIterator:


    def __init__(self, model: BaseActivationsModule, dataset: Dataset2D, o: PyTorchMeasureOptions,
                 activations_transformer: ActivationsTransformer = IdentityActivationsTransformer()):
        """
        Constructor for PytorchActivationsIterator.

        :param model: ActivationsModule
            The model to compute activations with.
        :param dataset: Dataset2D
            The dataset to compute activations from. The dataset should be a 2D dataset, where each sample is a pair of elements from the original dataset.
        :param o: PyTorchMeasureOptions
            The options for computing the measure.
        :param activations_transformer: ActivationsTransformer
            An optional transformer to apply to the activations after they have been computed.
        """
        self.model = model
        self.dataset = dataset
        self.o = o
        self.activations_transformer = activations_transformer

    def move_activations_to_measure_device(self,activations:ActivationValues):
        for a in activations.values():
            if self.o.model_device != self.o.measure_device:
                a=a.to(self.o.measure_device,non_blocking=True)

    def transform_activations(self,activations:ActivationValues,x_transformed,transformations)->ActivationValues:
        for k, a in activations.items():
            activations[k] = self.activations_transformer.transform(a, x_transformed,transformations)
        return activations

    def get_rows_cols(self,batch_i,x_transformed)->tuple[list[int],list[int]]:
        sample_i_start = batch_i*self.o.batch_size
        actual_batch_size = x_transformed.shape[0]
        i_samples = [self.dataset.d1tod2(i) for i in range(sample_i_start,sample_i_start+actual_batch_size)]
        return zip(*i_samples)


    def feed_batch(self, batch_i: int, x: torch.Tensor, tm: ThreadsComputationModel):
        i_rows, i_cols = self.get_rows_cols(batch_i, x)
        x = x.to(self.o.model_device, non_blocking=True)
        activations = self.model.forward_activations(x)
        transformations = self.dataset.get_transformations(i_rows, i_cols)
        self.move_activations_to_measure_device(activations)
        activations = self.transform_activations(activations, x, transformations)
        if tm.stop:
            return
        for row, row_activations in self.split_activations_by_row(activations, i_rows):
            tm.put(row, row_activations)

    @torch.no_grad
    def feed_measures(self,tm:ThreadsComputationModel):
        rows, cols = self.dataset.len0, self.dataset.len1
        logger.info(f"rows {rows} cols {cols}")
        dataloader = DataLoader(self.dataset, batch_size=self.o.batch_size, shuffle=False, num_workers=self.o.num_workers,pin_memory=True)

        if tm.stop:
            return
        for batch_i,x in tqdm.tqdm(enumerate(dataloader), disable=not self.o.verbose, leave=False):
            self.feed_batch(batch_i, x, tm)
            if tm.stop:
                return

    def split_activations_by_row(self,activations:ActivationValues,i_rows:list[int])->Generator[tuple[int,ActivationValues]]:
        all_rows = list(range(min(i_rows),max(i_rows)+1))
        start = 0
        last = all_rows[-1]
        logger.debug(f"rows: {all_rows}, last {last}")
        for current_row in all_rows:
            if current_row == last:
                end = len(i_rows)
            else:
                end = i_rows.index(current_row+1)
            activations_row = {k:a[start:end,] for k,a in activations.items()}
            # print(activations_row[0].shape,i_rows,start,end)
            start=end
            yield current_row,activations_row

    def evaluate(self, m: PyTorchActivationMeasure):
        activation_names = self.model.activation_names()
        rows, cols = self.dataset.len0, self.dataset.len1
        logger.debug(f"Main thread {threading.get_ident()}")
        measure_functions = {l:m.eval for l in activation_names}
        model_evaluating_function = self.feed_measures
        max_workers = len(activation_names)+1
        tm = ThreadsComputationModel(model_evaluating_function,measure_functions,max_workers,rows,cols,self.o.batch_size)
        return tm.execute()
