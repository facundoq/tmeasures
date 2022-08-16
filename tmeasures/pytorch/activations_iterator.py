from .dataset2d import STDataset, Dataset2D
import torch
from torch.utils.data import DataLoader
from . import ActivationsModule
from .base import PyTorchMeasureOptions, PyTorchLayerMeasure, PyTorchMeasure

from .activations_transformer import ActivationsTransformer
from ..utils.iterable_queue import IterableQueue
from .. import Transformation, InvertibleTransformation
import tqdm.auto as tqdm
import concurrent.futures

from typing import List

import abc


class ActivationsTransformer(abc.ABC):

    @abc.abstractmethod
    def transform(self, activations: torch.Tensor, x: torch.Tensor, transformations: List[Transformation]) -> torch.Tensor:
        pass


class IdentityActivationsTransformer(ActivationsTransformer):
    def transform(self, activations: torch.Tensor, x: torch.Tensor, transformations: List[Transformation]) -> torch.Tensor:
        return activations

def worker_callbacks(f):
    e = f.exception()

    if e is None:
        return

    trace = []
    tb = e.__traceback__
    while tb is not None:
        trace.append({
            "filename": tb.tb_frame.f_code.co_filename,
            "name": tb.tb_frame.f_code.co_name,
            "lineno": tb.tb_lineno
        })
        tb = tb.tb_next
    print(str({
        'type': type(e).__name__,
        'message': str(e),
        'trace': trace
    }))


class PytorchActivationsIterator:

    def __init__(self, model: ActivationsModule, dataset: Dataset2D, o: PyTorchMeasureOptions,
                 activations_transformer: ActivationsTransformer = IdentityActivationsTransformer()):
        self.model = model
        self.dataset = dataset
        self.o = o
        self.activations_transformer = activations_transformer

    def evaluate(self, m: PyTorchLayerMeasure):
        layers = self.model.activation_names()
        rows, cols = self.dataset.len0, self.dataset.len1
        prefix = f"{m.__class__.__name__}_"
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(layers),thread_name_prefix=prefix) as executor:

            qs = {l: IterableQueue(rows, maxsize=1,name=f"q({l})") for l in layers}

            threads = [executor.submit(m.eval,q,l) for q,l in zip(qs.values(),layers)]
            for t in threads:
                t.add_done_callback(worker_callbacks)

            

            with torch.no_grad():
                # print(f"act it starting,num workers {self.o.num_workers}:")
                for row in tqdm.trange(rows, disable=not self.o.verbose, leave=False):
                    row_dataset = self.dataset.row_dataset(row)
                    row_dataloader = DataLoader(row_dataset, batch_size=self.o.batch_size, shuffle=False, num_workers=0,pin_memory=True)
                    n_batch = len(row_dataloader)

                    row_qs = {l: IterableQueue(n_batch,maxsize=1,name=f"q({l}_{row})") for l in layers}

                    for k, q in qs.items():
                        # print(f"AI: putting row {row} dataloader for layer {k}")
                        q.put(row_qs[k])

                    # print(f"AI: finished putting row {row} dataloaders for all layers")
                    # for k,q in qs.items():
                    #     print(f"AI: {k}→ {q.queue.qsize()} items")


                    col = 0
                    # print("col",col)

                    for batch_i,x_transformed in enumerate(row_dataloader):
                        # print(f"AI: {batch_i}: moving to device {self.o.model_device}... ")
                        x_transformed = x_transformed.to(self.o.model_device,non_blocking=True)
                        # print("AI: getting activations..")
                        activations = self.model.forward_activations(x_transformed)
                        # print("AI: got activations")
                        col_to = col + x_transformed.shape[0]
                        for i, layer_activations in enumerate(activations):
                            if self.o.model_device != self.o.measure_device:
                                layer_activations=layer_activations.to(self.o.measure_device,non_blocking=True)
                            transformations = self.dataset.get_transformations(row, col, col_to)
                            layer_activations = self.activations_transformer.transform(layer_activations, x_transformed,transformations)
                            # print(f"AI: act it, shape {layer_activations.shape}")
                            # print(f"AI: putting col {col} batch for layer {i} ({layers[i]})")
                            row_qs[layers[i]].put(layer_activations)
                            # print(f"put {layer_activations.shape} into {layers[i]} {row_qs[layers[i]]}")
                        col = col_to
                        # print("AI: finished row")
                    # print("AI: finished all rows")
            results = [t.result() for t in threads]

        return results
