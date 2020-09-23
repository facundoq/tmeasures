#from __future__ import annotations
from typing import List
import numpy as np
import abc
import re

from transformational_measures.numpy.stats_running import RunningMeanWelford
from .utils import get_all

ActivationsByLayer = [np.ndarray]

class MeasureResult:
    def __init__(self,layers:ActivationsByLayer,layer_names:List[str],measure:'Measure',extra_values=dict()):
        # if len(layers) != len(layer_names):
            # print(len(layers),len(layer_names))
            # print(layer_names)
            # for l in layers:
                # print(l.shape)
        assert (len(layers) == len(layer_names))
        self.layers=layers
        self.layer_names=layer_names
        self.measure=measure
        self.extra_values=extra_values

    def __repr__(self):
        return f"MeasureResult({self.measure})"

    def all_1d(self):
        return np.any([ len(l.shape)==1 for l in self.layers])

    def per_layer_average(self) -> np.ndarray:
        result = []
        for layer in self.layers:
            layer=layer[:]
            layer_average=layer[np.isfinite(layer)].mean()
            result.append(layer_average)
        return np.array(result)

    def per_layer_mean_std(self) -> (np.ndarray,np.ndarray):
        means = []
        stds = []
        for layer in self.layers:
            layer=layer[:]
            layer_mean=layer[np.isfinite(layer)].mean()
            layer_std=layer[np.isfinite(layer)].std()
            means.append(layer_mean)
            stds.append(layer_std)
        return np.array(means),np.array(stds)

    def remove_layers(self,remove_indices:[int])->'MeasureResult':
        n = len(self.layer_names)
        all_indices=set(list(range(n)))
        keep_indices = list(all_indices.difference(set(remove_indices)))
        layers = get_all(self.layers,keep_indices)
        layer_names = get_all(self.layer_names,keep_indices)
        return MeasureResult(layers,layer_names,self.measure,self.extra_values)

    def weighted_global_average(self):
        return self.per_layer_average().mean()

    def global_average(self)-> float:
        rm = RunningMeanWelford()
        for layer in self.layers:
                for act in layer[:]:
                    if np.isfinite(act):
                        rm.update(act)
        return rm.mean()


class StratifiedMeasureResult(MeasureResult):
    def __init__(self, layers:ActivationsByLayer, layer_names:List[str], measure:'Measure', class_measures:List[MeasureResult], class_labels:List[str]):
        super().__init__(layers,layer_names,measure)
        self.class_measures=class_measures
        self.class_labels=class_labels

    def __repr__(self):
        return f"StratifiedMeasureResult {self.measure}"




class Measure(abc.ABC):
    def __repr__(self):
        return f"{self.__class__.__name__}"

    def id(self):
        return str(self)

    # @abc.abstractmethod
    # def name(self):
    #     pass
    #
    # @abc.abstractmethod
    # def abbreviation(self):
    #     pass

    # Returns the name of the measure, separated by spaces
    def name(self):
        # get first part before parenthesis (if any)
        class_name = self.__class__.__name__
        result = class_name.split("(")[0]
        # add a space before capitalized words
        result = re.sub( r"([A-Z])", r" \1", result)
        return result

    # generates an abbreviation of the measure, based on its name
    def abbreviation(self):
        initials=re.findall('[A-Z]', self.name())
        return "".join(initials)

    @abc.abstractmethod
    def eval(self)->MeasureResult:
        pass