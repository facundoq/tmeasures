from .dataset2d import STDataset, Dataset2D
import torch
from torch.utils.data import DataLoader
from . import ActivationsModule
from .base import PyTorchMeasureOptions, PyTorchLayerMeasure, PyTorchMeasure

from .activations_transformer import ActivationsTransformer
from ..utils.iterable_queue import IterableQueue
import queue
from .. import Transformation, InvertibleTransformation
import tqdm.auto as tqdm
import concurrent.futures

try:
    import namedthreads
    namedthreads.patch()
except ImportError: pass


from typing import List

import abc


class ActivationsTransformer(abc.ABC):

    @abc.abstractmethod
    def transform(self, activations: torch.Tensor, x: torch.Tensor, transformations: List[Transformation]) -> torch.Tensor:
        pass


class IdentityActivationsTransformer(ActivationsTransformer):
    def transform(self, activations: torch.Tensor, x: torch.Tensor, transformations: List[Transformation]) -> torch.Tensor:
        return activations

from tmeasures import logger
from concurrent.futures import ThreadPoolExecutor
import threading

class ThreadsManager:

    def __init__(self,layers:list[str],rows:int,n_batch:int,stop=False) -> None:
        self.stop=stop
        self.layers = layers
        self.qs = {l: IterableQueue(rows,maxsize=1,name=f"q({l})") for l in layers}
        self.row_qs = {l: IterableQueue(n_batch,maxsize=1,name=f"q({l}_row)") for l in layers}
    @property
    def queues(self):
        return list(self.qs.values())+list(self.row_qs.values())
    
    def cancel(self,server_future):
        logger.info("Exception raised when computing measure, shutting down all computing threads...")
        self.stop=True
        for q in self.queues:
            q.stop=True
        logger.info(f"Emptying queues to ensure server can move on and stop..")
        self.empty_all()
        # wait for server thread to finish
        concurrent.futures.wait([server_future], return_when=concurrent.futures.ALL_COMPLETED)
        # threading.Event().wait(0.1)
        logger.info("Server stopped. Pushing values to queues to ensure workers can consume and stop..")
        self.stop_all()
        logger.info("Workers stopped")
    
    def empty_all(self):
        for q in self.queues:
            try: q.get(block=False) 
            except queue.Empty: pass
 

    def stop_all(self):
        for q in self.queues:
            try: q.put(None,block=False)
            except queue.Full: pass
    

        

class PytorchActivationsIterator:

    def __init__(self, model: ActivationsModule, dataset: Dataset2D, o: PyTorchMeasureOptions,
                 activations_transformer: ActivationsTransformer = IdentityActivationsTransformer()):
        self.model = model
        self.dataset = dataset
        self.o = o
        self.activations_transformer = activations_transformer

    def check_finished(self,worker_futures,server_future,tm:ThreadsManager):
        futures = [server_future]+list(worker_futures.values())
        logger.info(f"Waiting for {len(worker_futures)} workers and server to finish...")
        done, not_done = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_EXCEPTION)
        logger.info(f"done")
        # check if an exception was raised
        
        if server_future in done:
            server_exception = server_future.exception()
        else:
            server_exception = None
        worker_exceptions = [f.exception() if f in done else None for f in worker_futures.values()]
        worker_failure = any([not e is None for e in worker_exceptions])
        if server_exception is None and not worker_failure:
            logger.info("No exceptions found, threads finished.")
            return None
        else: # we have some exceptions
            # signal stop for server
            logger.info("Some threads failed, terminating")
            tm.cancel(server_future)
            done, not_done = concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)

            if not server_exception is None:
                logger.info(f"Server exception, about to re raise from main thread\n{server_exception}\n")
                raise server_exception
            else:
                for e in worker_exceptions:
                    if not e is None:
                        logger.info(f"Worker exception, about to re raise from main thread\n{e}\n thread id {threading.get_ident()}\n")
                        raise e
                    
    def feed_threads(self,tm:ThreadsManager):
         layers = self.model.activation_names()
         rows, cols = self.dataset.len0, self.dataset.len1
         
         with torch.no_grad():
                # print(f"act it starting,num workers {self.o.num_workers}:")
                for row in tqdm.trange(rows, disable=not self.o.verbose, leave=False):
                    row_dataset = self.dataset.row_dataset(row)
                    row_dataloader = DataLoader(row_dataset, batch_size=self.o.batch_size, shuffle=False, num_workers=0,pin_memory=True)
                    
                    for k, q in tm.qs.items():
                        logger.info(f"AI: putting row {row} dataloader for  layer {k}")
                        q.put(tm.row_qs[k])

                    # print(f"AI: finished putting row {row} dataloaders for all layers")
                    # for k,q in qs.items():
                    #     print(f"AI: {k}→ {q.queue.qsize()} items")
                    if tm.stop:
                            logger.info("Server thread stopping, exception detected")
                            return
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
                            tm.row_qs[layers[i]].put(layer_activations)
                            # print(f"put {layer_activations.shape} into {layers[i]} {row_qs[layers[i]]}")
                            # Check if there's been an exception 
                            if tm.stop:
                                logger.info("Server thread stopping, exception detected")
                                return
                        col = col_to
                        # print("AI: finished row")
                    # print("AI: finished all rows")
   

    def evaluate(self, m: PyTorchLayerMeasure):
        self.queues=[]
        layers = self.model.activation_names()
        rows, cols = self.dataset.len0, self.dataset.len1
        prefix = f"{m.__class__.__name__}_"
        logger.info(f"Main thread {threading.get_ident()}")
        # calculate number of batches per row
        n_batch = cols // self.o.batch_size 
        if (cols % self.o.batch_size) >0:
            n_batch+=1

        
        with ThreadPoolExecutor(max_workers=len(layers)+1,thread_name_prefix=prefix) as executor:
            tm = ThreadsManager(layers,rows,n_batch,stop=False)
            logger.info("starting server thread")
            server_thread = executor.submit(self.feed_threads,tm)
            logger.info("starting worker threads")
            worker_threads = {l:executor.submit(m.eval,q,l) for l,q in tm.qs.items()}
            
            r = self.check_finished(worker_threads,server_thread,tm)
            
            if r is None:
                return [t.result() for t in worker_threads.values()]
            else:
                raise r 
