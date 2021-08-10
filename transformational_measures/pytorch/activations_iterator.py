from .dataset2d import STDataset, Dataset2D
import torch
from torch.utils.data import DataLoader
from . import ObservableLayersModule
from .base import PyTorchMeasureOptions, PyTorchLayerMeasure, PyTorchMeasure

from .activations_transformer import ActivationsTransformer
from ..utils.iterable_queue import IterableQueue
from .. import Transformation, InvertibleTransformation
import tqdm.auto as tqdm
import concurrent.futures
import typing

import abc


class ActivationsTransformer(abc.ABC):

    @abc.abstractmethod
    def transform(self, activations: torch.Tensor, x: torch.Tensor, transformations: [Transformation]) -> torch.Tensor:
        pass


class IdentityActivationsTransformer(ActivationsTransformer):
    def transform(self, activations: torch.Tensor, x: torch.Tensor, transformations: [Transformation]) -> torch.Tensor:
        return activations


class PytorchActivationsIterator:

    def __init__(self, model: ObservableLayersModule, dataset: Dataset2D, o: PyTorchMeasureOptions,
                 activations_transformer: ActivationsTransformer = IdentityActivationsTransformer()):
        self.model = model
        self.dataset = dataset
        self.o = o
        self.activations_transformer = activations_transformer

    def evaluate(self, m: PyTorchLayerMeasure):
        layers = self.model.activation_names()
        rows, cols = self.dataset.len0, self.dataset.len1

        with concurrent.futures.ThreadPoolExecutor() as executor:

            qs = {l: IterableQueue(rows, maxsize=1) for l in layers}

            threads = [executor.submit(m.eval, q) for q in qs.values()]

            self.model.eval()

            with torch.no_grad():

                for row in tqdm.trange(rows, disable=not self.o.verbose, leave=False):
                    row_dataset = self.dataset.row_dataset(row)
                    row_dataloader = DataLoader(row_dataset, batch_size=self.o.batch_size, shuffle=False,
                                                num_workers=self.o.num_workers,pin_memory=True)
                    n_batch = len(row_dataloader)

                    row_qs = {l: IterableQueue(n_batch) for l in layers}
                    for k, q in qs.items():
                        q.put(row_qs[k])
                    col = 0
                    for x_transformed in row_dataloader:
                        x_transformed = x_transformed.to(self.o.model_device,non_blocking=True)
                        y, activations = self.model.forward_intermediates(x_transformed)
                        col_to = col + x_transformed.shape[0]
                        for i, layer_activations in enumerate(activations):
                            if self.o.model_device != self.o.measure_device:
                                layer_activations=layer_activations.to(self.o.measure_device,non_blocking=True)
                            transformations = self.dataset.get_transformations(row, col, col_to)
                            layer_activations = self.activations_transformer.transform(layer_activations, x_transformed,
                                                                                       transformations)
                            # print(f"act it, shape {layer_activations.shape}")

                            row_qs[layers[i]].put(layer_activations)
                            # print(f"put {layer_activations.shape} into {layers[i]} {row_qs[layers[i]]}")
                        col = col_to

            results = [t.result() for t in threads]

        return results
