#from __future__ import annotations
import abc
from typing import List
from tmeasures.transformation import TransformationSet

import numpy as np


class ActivationsIterator(abc.ABC):
    """
    .. 
        Iterate over the ST matrix of activations of a network, varying the samples and transformations.
        The iteration is in both orders:
        1) transformations_first:  column indexes first, then row indices indices
                For transformation 0, get all samples (col 1, all rows), then (col 2, all rows)
                The iteration over rows in done by batches.
        2) samples_first: row indexes first, then column
                Same as transformations_first but transposed.

    """

    @abc.abstractmethod
    def get_transformations(self)->TransformationSet:
        pass

    @abc.abstractmethod
    def transformations_first(self):
        pass

    @abc.abstractmethod
    def samples_first(self):
        pass

    @abc.abstractmethod
    def layer_names(self) -> List[str]:
        pass

    @abc.abstractmethod
    def get_inverted_activations_iterator(self) -> 'ActivationsIterator':
        pass

    @abc.abstractmethod
    def get_both_iterator(self) -> 'ActivationsIterator':
        pass

    @abc.abstractmethod
    def get_normal_activations_iterator(self) -> 'ActivationsIterator':
        pass

    def row_from_iterator(self,transformation_activations_iterator):
        '''
        Get a row of the ST matrix from the :param transformation_activations_iterator
        :return: row of the ST matrix with the  activations for all the transformations of sample, and also the transformed samples
        '''
        activations=[[] for i in range(len(self.layer_names()))]
        x_transformed=[]
        for  x_transformed_batch,activations_batch in transformation_activations_iterator:
            x_transformed.append(x_transformed_batch)
            for i, layer_activations in enumerate(activations_batch):
                activations[i].append(layer_activations)
        activations = [ np.concatenate(layer_activations,axis=0) for layer_activations in activations]
        x_transformed = np.concatenate(x_transformed,axis=0)
        return activations,x_transformed



