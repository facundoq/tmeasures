

from .activations_iterator import ActivationsIterator
import numpy as np
from typing import  List
from .. import MeasureResult, StratifiedMeasureResult
from ..measure import Measure

import abc


class NumpyMeasure(Measure):

    @abc.abstractmethod
    def eval(self,activations_iterator:ActivationsIterator,verbose=False)-> MeasureResult:
        '''

        '''
        pass


    def eval_stratified(self,classes_iterators:List[ActivationsIterator],class_labels:List[str])-> StratifiedMeasureResult:
        '''
        Calculate the `variance_measure` for each class separately
        Also calculate the average stratified `variance_measure` over all classes
        '''
        variance_per_class = [self.eval(iterator) for iterator in classes_iterators]
        stratified_measure_layers = self.mean_variance_over_classes(variance_per_class)

        layer_names= variance_per_class[0].layer_names

        return StratifiedMeasureResult(stratified_measure_layers, layer_names, self, variance_per_class, class_labels)


    def mean_variance_over_classes(self, class_variance_result:List[MeasureResult]) -> List[np.ndarray]:
        # calculate the mean activation of each unit in each layer over the set of classes
        class_variance_layers=[r.layers for r in class_variance_result]
        # class_variance_layers is a list (classes) of list layers)
        # turn it into a list (layers) of lists (classes)
        layer_class_vars=[list(i) for i in zip(*class_variance_layers)]
        # compute average variance of each layer over classses
        layer_vars=[ sum(layer_values)/len(layer_values) for layer_values in layer_class_vars]
        return layer_vars

