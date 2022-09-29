from __future__ import annotations

from typing import List
from .base import NumpyMeasure
from .. import MeasureResult
from .activations_iterator import ActivationsIterator
from .stats_running import RunningMeanWelford
import numpy as np

def list_get_all(list:List,indices:List[int])->List:
    return [list[i] for i in indices]

class DistanceFunction:
    def __init__(self,normalize:bool):
        self.normalize=normalize

    def distance(self,batch:np.ndarray,batch_inverted:np.ndarray,mean_running:RunningMeanWelford):
        n_shape=len(batch.shape)
        assert n_shape>=2
        n,f=batch.shape[0],batch.shape[1]

        if n_shape>2 and self.normalize:
            # normalize all extra dimensions
            for i in range(n):
                for j in range(f):
                    batch[i,j,:]/=np.linalg.norm(batch[i,j,:])
                    batch_inverted[i,j,:]/=np.linalg.norm(batch_inverted[i,j,:])

        # ssd of all values
        distances = (batch-batch_inverted)**2
        n_shape=len(batch.shape)
        if n_shape>2:
            # aggregate extra dims to keep only the feature dim
            distances= distances.mean(axis=tuple(range(2,n_shape)))
        distances = np.sqrt(distances)
        assert len(distances.shape)==2
        mean_running.update_all(distances)
    def __repr__(self):
        return f"DF(normalize={self.normalize})"

class DistanceSameEquivarianceSimple(NumpyMeasure):
    def __init__(self, distance_function:DistanceFunction):
        super().__init__()
        self.distance_function=distance_function

    def __repr__(self):
        return f"DSES(df={self.distance_function})"

    def name(self)->str:
        return "Distance Same-Equivariance Simple"
    def abbreviation(self):
        return "DSES"

    def eval(self,activations_iterator:ActivationsIterator,verbose=False)->MeasureResult:
        activations_iterator = activations_iterator.get_both_iterator()
        mean_running=None

        for x, transformation_activations_iterator in activations_iterator.samples_first():
            # transformation_activations_iterator can iterate over all transforms
            for x_transformed, activations,inverted_activations in transformation_activations_iterator:
                if mean_running is None:
                    # do this after the first iteration since we dont know the number
                    # of layers until the first iteration of the activations_iterator
                    mean_running = [RunningMeanWelford() for i in range(len(activations))]
                for j, (layer_activations,inverted_layer_activations) in enumerate(zip(activations,inverted_activations)):
                    self.distance_function.distance(layer_activations,inverted_layer_activations,mean_running[j])
        # calculate the final mean over all samples (and layers)
        means = [b.mean() for b in mean_running]
        return MeasureResult(means,activations_iterator.layer_names(),self)


