from collections.abc import Generator
import typing

from .threads_manager import ThreadsManager

from .transformations import PyTorchTransformation
from .dataset2d import STDataset, Dataset2D
import torch
from torch.utils.data import DataLoader
from . import ActivationsModule
from .base import PyTorchMeasureOptions, PyTorchLayerMeasure, PyTorchMeasure

# from .activations_transformer import ActivationsTransformer
from .. import Transformation, InvertibleTransformation
import tqdm.auto as tqdm
import concurrent.futures

try:
    import namedthreads
    namedthreads.patch()
except ImportError: pass


from typing import Callable, List

import abc


class ActivationsTransformer(abc.ABC):

    @abc.abstractmethod
    def transform(self, activations: torch.Tensor, x: torch.Tensor, transformations: List[PyTorchTransformation]) -> torch.Tensor:
        pass


class IdentityActivationsTransformer(ActivationsTransformer):
    def transform(self, activations: torch.Tensor, x: torch.Tensor, transformations: List[PyTorchTransformation]) -> torch.Tensor:
        return activations

from .. import logger as tm_logger
logger = tm_logger.getChild("pytorch.activations_iterator")
from concurrent.futures import ThreadPoolExecutor
import threading

class PytorchActivationsIterator:


    def __init__(self, model: ActivationsModule, dataset: Dataset2D, o: PyTorchMeasureOptions,
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
    
    def move_activations_to_measure_device(self,activations:list[torch.Tensor]):
        for i, layer_activations in enumerate(activations):
                if self.o.model_device != self.o.measure_device:
                    layer_activations=layer_activations.to(self.o.measure_device,non_blocking=True)

    def transform_activations(self,activations:list[torch.Tensor],x_transformed,transformations)->list[torch.Tensor]:
        for i, layer_activations in enumerate(activations):
            activations[i] = self.activations_transformer.transform(layer_activations, x_transformed,transformations)
        return activations

    def get_rows_cols(self,batch_i,x_transformed)->tuple[list[int],list[int]]:
        sample_i_start = batch_i*self.o.batch_size
        actual_batch_size = x_transformed.shape[0]
        i_samples = [self.dataset.d1tod2(i) for i in range(sample_i_start,sample_i_start+actual_batch_size)]
        return zip(*i_samples)
    
            
    @torch.no_grad
    def feed_threads2(self,tm:ThreadsManager):
        layers = self.model.activation_names()
        rows, cols = self.dataset.len0, self.dataset.len1
        logger.info(f"rows {rows} cols {cols}")
        # print(f"act it starting,num workers {self.o.num_workers}:")
        dataloader = DataLoader(self.dataset, batch_size=self.o.batch_size, shuffle=False, num_workers=self.o.num_workers,pin_memory=True)
        i=0
        # print(f"AI: finished putting row {row} dataloaders for all layers")
        # for k,q in qs.items():
        #     print(f"AI: {k}→ {q.queue.qsize()} items")
        if tm.stop:
            logger.info("Server thread stopping, exception detected")
            return
            # print("col",col)
        for batch_i,x_transformed in tqdm.tqdm(enumerate(dataloader), disable=not self.o.verbose, leave=False):
            i_rows,i_cols = self.get_rows_cols(batch_i,x_transformed)
            logger.info(f"Rows/cols {i_rows}, {i_cols}")
            # print(f"AI: {batch_i}: moving to device {self.o.model_device}... ")
            x_transformed = x_transformed.to(self.o.model_device,non_blocking=True)
            # print("AI: getting activations..")
            activations = self.model.forward_activations(x_transformed)
            # print("AI: got activations")

            transformations = self.dataset.get_transformations(i_rows,i_cols)
            
            self.move_activations_to_measure_device(activations)
            activations = self.transform_activations(activations,x_transformed,transformations)
            if tm.stop:
                logger.info("Server thread stopping, exception detected")
                return
                # print(f"AI: act it, shape {layer_activations.shape}")
                # print(f"AI: putting col {col} batch for layer {i} ({layers[i]})")
            for row, row_activations in self.split_activations_by_row(activations,i_rows):
                tm.put(row,row_activations)
            if tm.stop:
                logger.info("Server thread stopping, exception detected")
                return
                # print("AI: finished row")
            # print("AI: finished all rows")

    def split_activations_by_row(self,activations:list[torch.Tensor],i_rows:list[int])->Generator[tuple[int,dict[str,torch.Tensor]]]:
        layers = self.model.activation_names()
        all_rows = list(range(min(i_rows),max(i_rows)+1))
        start = 0
        last = all_rows[-1]
        logger.info(f"rows: {all_rows}, last {last}")
        for current_row in all_rows:
            if current_row == last:
                end = len(i_rows)
            else:
                end = i_rows.index(current_row+1)
            activations_row = [a[start:end,] for a in activations]
            activations_row_dict = { layers[i]:a for i,a in enumerate(activations_row)}
            # print(activations_row[0].shape,i_rows,start,end)
            start=end
            yield current_row,activations_row_dict


    # @torch.no_grad
    # def feed_threads(self,tm:ThreadsManager):
    #     layers = self.model.activation_names()
    #     rows, cols = self.dataset.len0, self.dataset.len1

    #     # print(f"act it starting,num workers {self.o.num_workers}:")
    #     for row in tqdm.trange(rows, disable=not self.o.verbose, leave=False):
    #         row_dataset = self.dataset.row_dataset(row)
    #         row_dataloader = DataLoader(row_dataset, batch_size=self.o.batch_size, shuffle=False, num_workers=0,pin_memory=True)
            
    #         for k, q in tm.qs.items():
    #             logger.info(f"AI: putting row {row} dataloader for  layer {k}")
    #             q.put(tm.row_qs[k])

    #         # print(f"AI: finished putting row {row} dataloaders for all layers")
    #         # for k,q in qs.items():
    #         #     print(f"AI: {k}→ {q.queue.qsize()} items")
    #         if tm.stop:
    #                 logger.info("Server thread stopping, exception detected")
    #                 return
    #         col = 0
    #         # print("col",col)
            
    #         for batch_i,x_transformed in enumerate(row_dataloader):
    #             # print(f"AI: {batch_i}: moving to device {self.o.model_device}... ")
    #             x_transformed = x_transformed.to(self.o.model_device,non_blocking=True)
    #             # print("AI: getting activations..")
    #             activations = self.model.forward_activations(x_transformed)
    #             # print("AI: got activations")
                
    #             n_batch = x_transformed.shape[0]
    #             col_to = col + n_batch
    #             i_rows = [row]*n_batch
    #             i_cols = list(range(col,col_to))
    #             logger.info("Rows/cols",i_rows,i_cols)
    #             transformations = self.dataset.get_transformations(i_rows,i_cols)

    #             for i, layer_activations in enumerate(activations):
    #                 if self.o.model_device != self.o.measure_device:
    #                     layer_activations=layer_activations.to(self.o.measure_device,non_blocking=True)

                    
                    
    #                 layer_activations = self.activations_transformer.transform(layer_activations, x_transformed,transformations)
    #                 # print(f"AI: act it, shape {layer_activations.shape}")
    #                 # print(f"AI: putting col {col} batch for layer {i} ({layers[i]})")
    #                 tm.row_qs[layers[i]].put(layer_activations)
    #                 # print(f"put {layer_activations.shape} into {layers[i]} {row_qs[layers[i]]}")
    #                 # Check if there's been an exception 
    #                 if tm.stop:
    #                     logger.info("Server thread stopping, exception detected")
    #                     return
    #             col = col_to
    #             # print("AI: finished row")
    #         # print("AI: finished all rows")

    def evaluate(self, m: PyTorchLayerMeasure):
        layers = self.model.activation_names()
        rows, cols = self.dataset.len0, self.dataset.len1
        prefix = f"{m.__class__.__name__}_"
        logger.info(f"Main thread {threading.get_ident()}")
        # calculate number of batches per row
        # if batch_size > cols, then note that n_batch = 1
        measure_functions = {l:m.eval for l in layers}
        model_evaluating_function = self.feed_threads2
        max_workers = len(layers)+1
        tm = ThreadsManager(model_evaluating_function,measure_functions,max_workers,rows,cols,self.o.batch_size)
        return tm.execute()