from typing import List
from ..base import NumpyMeasure
from tmeasures import MeasureResult
from ..activations_iterator import ActivationsIterator

import multiprocessing
from queue import Queue
from threading import Thread

import abc

class LayerMeasure(abc.ABC):
    def __init__(self,id:int,name:str):
        self.id=id
        self.name=name

    def eval_private(self,q:Queue,inner_q:Queue,rq:Queue):
        result=self.eval(q,inner_q)
        rq.put(result)


    @abc.abstractmethod
    def eval(self,q:Queue,inner_q:Queue):
        pass

    def queue_as_generator(self,q: Queue):
        while True:
            v = q.get()
            if v is None:
                break
            else:
                yield v

from enum import Enum

class ActivationsOrder(Enum):
    TransformationsFirst = "tf"
    SamplesFirst = "sf"

class PerLayerMeasure(NumpyMeasure, abc.ABC):

    def __init__(self,activations_order:ActivationsOrder,queue_max_size=1,multiprocess=False):
        if multiprocess:
            print("Warning, deadlocks when using multiprocess are possible")
        self.activations_order = activations_order
        self.queue_max_size=queue_max_size
        if multiprocess:
            self.process_class = multiprocessing.Process
            self.queue_class = multiprocessing.Queue
        else:
            self.process_class = Thread
            self.queue_class = Queue

    @abc.abstractmethod
    def generate_layer_measure(self, i:int, name:str) -> LayerMeasure:
        pass

    def signal_iteration_end(self, queues: List[Queue]):
        self.put_value(queues,None)

    def put_values(self,queues: List[Queue],values:List):
        for q,v in zip(queues,values):
            q.put(v)

    def put_value(self,queues:List[Queue],value):
        for q in queues:
            q.put(value)

    def start_threads(self,threads:List):
        for t in threads:
            t.start()
    def wait_for_threads(self, threads: List):
        for t in threads:
            t.join()

    def eval(self,activations_iterator:ActivationsIterator)->MeasureResult:
        names = activations_iterator.layer_names()
        layers = len(names)
        layer_measures = [self.generate_layer_measure(i, name) for i, name in enumerate(names)]

        queues = [self.queue_class(self.queue_max_size) for i in range(layers)]
        inner_queues = [self.queue_class(self.queue_max_size) for i in range(layers)]
        result_queues = [self.queue_class(self.queue_max_size) for i in range(layers)]

        threads = [self.process_class(target=c.eval_private, args=[q, qi,qr],daemon=True) for c, q, qi, qr in
                   zip(layer_measures, queues, inner_queues,result_queues )]

        self.start_threads(threads)
        if self.activations_order == ActivationsOrder.SamplesFirst:
            self.eval_samples_first(activations_iterator,queues,inner_queues)
        elif self.activations_order == ActivationsOrder.TransformationsFirst:
            self.eval_transformations_first(activations_iterator, queues, inner_queues)
        else:
            raise ValueError(f"Unknown activations order {self.activations_order}")
        self.wait_for_threads(threads)
        results  = [qr.get() for qr in result_queues]
        return self.generate_result_from_layer_results(results,names)


    def generate_result_from_layer_results(self,results,names):
        return MeasureResult(results, names, self)

    def eval_samples_first(self,activations_iterator:ActivationsIterator, queues:List[Queue], inner_queues:List[Queue]):

        for activations, x_transformed in activations_iterator.samples_first():
            self.put_value(queues, x_transformed)
            self.put_values(inner_queues,activations)
            self.signal_iteration_end(inner_queues)
        self.signal_iteration_end(queues)


    def eval_transformations_first(self, activations_iterator: ActivationsIterator, queues: List[Queue],inner_queues: List[Queue]):

        for transformation, batch_activations in activations_iterator.transformations_first():
            self.put_value(queues,transformation)
            for x, batch_activation in batch_activations:
                self.put_values(inner_queues,batch_activation)
            self.signal_iteration_end(inner_queues)
        self.signal_iteration_end(queues)


class SamplesFirstPerLayerMeasure(PerLayerMeasure):
    def __init__(self):
        super().__init__(ActivationsOrder.SamplesFirst)

class TransformationsFirstPerLayerMeasure(PerLayerMeasure):
    def __init__(self):
        super().__init__(ActivationsOrder.TransformationsFirst)