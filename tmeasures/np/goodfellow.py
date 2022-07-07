from typing import Dict, List
from tmeasures import MeasureResult
from .activations_iterator import ActivationsIterator
from . import NumpyMeasure

from multiprocessing import Queue
from .quotient import divide_activations
from .multithread.multithreaded_layer_measure import LayerMeasure,PerLayerMeasure,ActivationsOrder
import numpy as np
from .stats_running import RunningMeanWelford

class GoodfellowGlobalInvariance(LayerMeasure):

    def __init__(self, id:int, name:str, activation_percentage:float,sign:int):
        super().__init__(id,name)
        self.activation_percentage = activation_percentage
        self.sign=sign

    def eval(self,q:Queue,inner_q:Queue):

        max_samples_per_transformation=20
        history=[]
        print(f"starting {self.id}")
        for transformation in self.queue_as_generator(q):
            i=0
            for activations in self.queue_as_generator(inner_q):
                if i<max_samples_per_transformation:
                    history.append(activations)
                i+=activations.shape[0]

        print(f"finish{self.id}")
        # sort all activation values for the different samples
        activations = np.vstack(history)
        del history
        if self.sign != 1:
            activations*=self.sign
        print(f"shape {activations.shape}")
        activations.sort(axis=0)
        n = activations.shape[0]
        # calculate the threshold indexes so that G(i) = activation_percentage
        threshold_indexes = int((1-self.activation_percentage)*n)
        # calculate the threshold values (approximately)
        self.thresholds = activations[threshold_indexes, :]

        # set g(i) equal to the activations_percentage
        self.g= np.zeros_like(self.thresholds) + self.activation_percentage


    def get_final_result(self):
        return self.g

class GoodfellowLocalInvariance(LayerMeasure):

    def __init__(self, id:int, name:str, threshold:float,sign:int):
        super().__init__(id,name)
        self.threshold = threshold
        self.sign=sign

    def eval(self,q:Queue,inner_q:Queue):
        running_mean = RunningMeanWelford()
        # activation_sum=0
        n=0
        for transformation in self.queue_as_generator(q):
            for activations in self.queue_as_generator(inner_q):
                if self.sign != 1:
                    activations *= self.sign
                activated = (activations > self.threshold) * 1
                running_mean.update_all(activated)

        self.l=running_mean.mean()

    def get_final_result(self):
        return self.l


class GlobalFiringRateMeasure(PerLayerMeasure):
    def __init__(self,activation_percentage:float,sign:int):
        super().__init__(ActivationsOrder.TransformationsFirst)
        self.activation_percentage:float=activation_percentage
        self.sign:int=sign
        self.layer_measures:Dict[int, GoodfellowGlobalInvariance]={}

    def __repr__(self):
        return f"G()"

    def generate_layer_measure(self, i:int, name:str) -> LayerMeasure:
        layer_measure = GoodfellowGlobalInvariance(i, name, self.activation_percentage, self.sign)
        self.layer_measures[i]=layer_measure
        return layer_measure

    def get_layer_measure(self,i:int):
        return self.layer_measures[i]

    def get_thresholds(self):
        return {i:m.thresholds  for i,m in self.layer_measures.items()}

class LocalFiringRateMeasure(PerLayerMeasure):
    def __init__(self,thresholds:List[np.ndarray],sign:int):
        super().__init__(ActivationsOrder.SamplesFirst)
        self.sign=sign
        self.thresholds=thresholds

    def __repr__(self):
        return f"L()"

    def generate_layer_measure(self, i:int, name:str) -> LayerMeasure:

        return GoodfellowLocalInvariance(i, name, self.thresholds[i], self.sign)

class GoodfellowInvariance(NumpyMeasure):


    def __init__(self,activations_percentage=0.01,sign=1):
        assert sign in [1,-1]
        super().__init__()
        self.activations_percentage=activations_percentage
        self.sign=sign

    def eval(self,activations_iterator:ActivationsIterator):
        print("calculating global")
        self.g = GlobalFiringRateMeasure(self.activations_percentage,self.sign)
        g_result = self.g.eval(activations_iterator)
        print("calculating thresholds")
        self.thresholds = self.g.get_thresholds()
        print("calculating local")
        self.l = LocalFiringRateMeasure(self.thresholds,self.sign)
        l_result = self.l.eval(activations_iterator)

        ratio = divide_activations(l_result.layers,g_result.layers)
        return MeasureResult(ratio, activations_iterator.layer_names(), self)

    def __repr__(self):
        return f"Goodfellow(gp={self.activations_percentage})"

    def name(self):
        return "Goodfellow"
    def abbreviation(self):
        return "GF"