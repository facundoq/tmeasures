import scipy.spatial.distance
import numpy as np
from abc import ABC,abstractmethod

class ActivationsTransformation(ABC):

    @abstractmethod
    def apply(self,activations:np.ndarray):
        pass

class DistanceAggregation(ActivationsTransformation):

    def __init__(self, normalize:bool, keep_shape:bool, distance="euclidean"):
        self.normalize=normalize
        self.keep_shape=keep_shape
        self.distance=distance


    def __repr__(self):
        return f"DA(normalize={self.normalize},keep_shape={self.keep_shape},distance={self.distance})"

    def normalize_activations(self,x:np.ndarray):
        c,n,d=x.shape
        if self.normalize and d>1:
            for i in range(c):
                for j in range(n):
                    x[i,j,:]/=np.linalg.norm(x[i,j,:])

    def apply(self,x:np.ndarray):
        x = self.convert_to_cnd_format(x)
        # x has size (c,n,d), where c is the feature dimension
        self.normalize_activations(x)
        return self.aggregate_distances(x)

    def convert_to_cnd_format(self,x:np.ndarray):
        l = len(x.shape)
        # Convert x to shape (n, features, dim_features)
        if l > 2:
            n, features, = x.shape[0:2]
            if self.keep_shape:
                # consider dims 2+ as a single dim
                # dim 0 is the number
                x = x.reshape((n, features, -1))

            else:
                # consider every element of the feature map as a distinct activation
                x = x.reshape((n, -1, 1))
        elif l == 2:
            n,c=x.shape
            x = x.reshape((n, c, 1))
        else:
            raise ValueError(f"Activation shape not supported {x.shape}")
        # ncd to cnd
        x = x.transpose((1, 0, 2))
        x = np.ascontiguousarray(x)
        return x

    def aggregate_distances(self, x: np.ndarray):
        c,n, d = x.shape
        results = np.zeros(c)
        for i in range(c):
            sample = x[i, :,:]
            dm = scipy.spatial.distance.pdist(sample, self.distance)
            results[i]= dm.mean()
        return results


