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

def worker_callbacks(f):
    return
    e = f.exception()

    if e is None:
        return
    raise e

    trace = []
    tb = e.__traceback__
    # while tb is not None:
    #     trace.append({
    #         "filename": tb.tb_frame.f_code.co_filename,
    #         "name": tb.tb_frame.f_code.co_name,
    #         "lineno": tb.tb_lineno
    #     })
    #     tb = tb.tb_next
    # print(str({
    #     'type': type(e).__name__,
    #     'message': str(e),
    #     'trace': trace
    # }),flush=True)


from tmeasures import logger
from concurrent.futures import ThreadPoolExecutor

class ExceptionAwareThreadPoolExecutor(concurrent.futures.ThreadPoolExecutor):
    def submit(self, fn, *args, on_exception='console', **kwargs):
        self.on_exception = on_exception
        return super().submit(self._function_wrapper, fn, *args, **kwargs)
        
    def _function_wrapper(self, fn, *args, **kwargs):
        try:
            # print(f"Starting {threading.get_native_id()}",flush=True)
            return fn(*args, **kwargs)
        except BaseException as e:
            if self.on_exception == 'console':
                # print stack to console:
                #logging.error(f'Exception class {e.__class__.__name__} raised', exc_info=True)
                raise e
            elif self.on_exception != None:
                self.on_exception(e)
            raise e
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
    
    def cancel(self):
        logger.error("Exception raised when computing measure, shutting down all computing threads...")
        for q in self.queues:
            q.stop=True
        logger.info("Emptying queues to ensure server can move on and stop..")
        self.empty_all()
        logger.info("Pushing values to queues to ensure workers can consume and stop..")
        self.stop_all()
    
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
        print(f"Waiting for {len(worker_futures)} workers and server to finish...",flush=True)
        done, not_done = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_EXCEPTION)
        print(f"done",flush=True)
        # check if an exception was raised
        
        if server_future in done:
            server_exception = server_future.exception()
        else:
            server_exception = None
        
        print("after1",flush=True)
        worker_exceptions = [f.exception() if f in done else None for f in worker_futures.values()]
        print("after2",flush=True)
        worker_failure = any([not e is None for e in worker_exceptions])
        print("after3",flush=True)
        if server_exception is None and not worker_failure:
            print("No exceptions found, threads terminated ok",
                  flush=True)
            return None
        else: # we have some exceptions
            # signal stop for server
            print("Some threads failed, terminating",flush=True)
            tm.cancel()
            done, not_done = concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)
            if not server_exception is None:
                print(f"about to re raise server exception from main thread\n{server_exception}\n",flush=True)
                raise server_exception
            else:
                for e in worker_exceptions:
                    if not e is None:
                        print(f"about to re raise worker exception from main thread\n{e}\n thread id {threading.get_ident()}\n",flush=True)
                        raise e
            
            # # shutdown server
            # if server_exception is None:
            #     # if server thread is running, finish it
            #     tm.empty_all()
            #     # flush all queues to ensure server can advance its loops
            # # shutdown workers
            # for l,wf in worker_futures.items():
            #     if wf.exception() is None:
            #         tm.stop_queue(l)
                    
            
            # for future in done:
            #     if future.exception() != None:
            #         logger.error(f"Exception in thread")
            #         callback()
            #         logger.error(f"raising exception in current thread {threading.get_ident()}")
            #         # print(future.exception())
            #         # raise future.exception()
            #         return future.exception()
        
    def feed_threads(self,tm:ThreadsManager):
         layers = self.model.activation_names()
         rows, cols = self.dataset.len0, self.dataset.len1
        
         with torch.no_grad():
                # print(f"act it starting,num workers {self.o.num_workers}:")
                for row in tqdm.trange(rows, disable=not self.o.verbose, leave=False):
                    row_dataset = self.dataset.row_dataset(row)
                    row_dataloader = DataLoader(row_dataset, batch_size=self.o.batch_size, shuffle=False, num_workers=0,pin_memory=True)
                    # n_batch = len(row_dataloader)
                    # row_qs = {l: IterableQueue(n_batch,maxsize=1,name=f"q({l}_{row})") for l in layers}
                    
                    for k, q in tm.qs.items():
                        # print(f"AI: putting row {row} dataloader for layer {k}")
                        q.put(tm.row_qs[k])

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
                            tm.row_qs[layers[i]].put(layer_activations)
                            # print(f"put {layer_activations.shape} into {layers[i]} {row_qs[layers[i]]}")
                            # Check if there's been an exception 
                            if tm.stop:
                                return
                        col = col_to
                        # print("AI: finished row")
                    # print("AI: finished all rows")
   

    def evaluate(self, m: PyTorchLayerMeasure):
        self.queues=[]
        layers = self.model.activation_names()
        rows, cols = self.dataset.len0, self.dataset.len1
        prefix = f"{m.__class__.__name__}_"
        logger.error(f"Main thread {threading.get_ident()}")
        # calculate number of batches per row
        n_batch = cols // self.o.batch_size 
        if (cols % self.o.batch_size) >0:
            n_batch+=1

        
        with ThreadPoolExecutor(max_workers=len(layers)+1,thread_name_prefix=prefix) as executor:
            tm = ThreadsManager(layers,rows,n_batch,stop=False)
            print("Submitting threads.")
            logger.info("starting server thread")
            server_thread = executor.submit(self.feed_threads,tm)
            logger.info("starting worker threads")
            worker_threads = {l:executor.submit(m.eval,q,l) for l,q in tm.qs.items()}
            
            # for t in threads:
            #     t.add_done_callback(worker_callbacks)            
            # self.feed_threads(qs)

            r = self.check_finished(worker_threads,server_thread,tm)
            if r is None:
                return [t.result() for t in worker_threads.values()]
            else:
                raise r 
