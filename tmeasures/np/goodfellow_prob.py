from typing import List
from tmeasures import MeasureResult
from .activations_iterator import ActivationsIterator
from . import NumpyMeasure
from .quotient import divide_activations

from multiprocessing import Queue
from .multithread.multithreaded_layer_measure import LayerMeasure,PerLayerMeasure,ActivationsOrder
import numpy as np
from .stats_running import RunningMeanAndVarianceWelford,RunningMeanWelford
from scipy.stats import norm
from tqdm import tqdm

default_alpha=0.99
default_sign=1

class GoodfellowNormalGlobalInvariance(NumpyMeasure):
    thresholds_key="thresholds"

    def __repr__(self):
        return f"{self.__class__.__name__}(gp={self.alpha})"

    # def name(self):
    #     return "GoodfellowNormalGlobal"
    #
    # def abbreviation(self):
    #     return "GGN"

    def __init__(self, alpha:float=default_alpha, sign:int=default_sign):
        super().__init__()
        self.alpha = alpha
        self.sign=sign

    def eval(self,activations_iterator: ActivationsIterator,verbose=False)->MeasureResult:
        running_means = [RunningMeanAndVarianceWelford() for i in activations_iterator.layer_names()]

        for transformation, samples_activations_iterator in tqdm(activations_iterator.transformations_first(),disable=not verbose):
            for x, batch_activations in samples_activations_iterator:
                for j, activations in enumerate(batch_activations):
                    if self.sign != 1: activations *= self.sign
                    running_means[j].update_all(activations)

        stds  = [running_mean.std() for running_mean in running_means]
        means = [running_mean.mean() for running_mean in running_means]
        original_shapes = [mean.shape for mean in means]
        means = [mean.reshape(mean.size) for mean in means]
        stds =  [std.reshape(std.size) for std in stds]
    # calculate the threshold values (approximately)
        thresholds=[np.zeros(mean.size) for mean in means]

        for i,(mean,std) in enumerate(zip(means,stds)):
            for j,(mu,sigma) in enumerate(zip(mean,std)):
                if sigma>0:
                    t=norm.ppf(self.alpha,loc=mu,scale=sigma)
                else:
                    t=mu
                thresholds[i][j]=t
    #thresholds = mean+2*std
        thresholds=[threshold.reshape(original_shape) for threshold,original_shape in zip(thresholds,original_shapes)]
        # set g(i) equal to the activations_percentage
        layers_g= [np.zeros_like(threshold) + (1-self.alpha) for threshold in thresholds]

        return MeasureResult(layers_g, activations_iterator.layer_names(), self,extra_values={self.thresholds_key:thresholds})

class GoodfellowNormalLocalInvariance(NumpyMeasure):

    def __init__(self, thresholds:List[np.ndarray],sign:int=default_sign):
        super().__init__()
        self.thresholds = thresholds
        self.sign=sign

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def eval(self,activations_iterator: ActivationsIterator,verbose=False)->MeasureResult:
        running_means = [RunningMeanWelford() for i in activations_iterator.layer_names()]

        for x,transformation_activations  in tqdm(activations_iterator.samples_first(),disable=not verbose):
            for x_transformed, activations in transformation_activations:
                for i, layer_activations in enumerate(activations):

                    if self.sign != 1:
                        layer_activations *= self.sign

                    activated:np.ndarray = (layer_activations > self.thresholds[i]) * 1.0
                    # print(activated.shape,activated.min(),activated.max(),activated.dtype)
                    if np.any(activated<0):
                        print(activated)
                    running_means[i].update_all(activated)

        layers_l = [m.mean() for m in running_means]

        return MeasureResult(layers_l, activations_iterator.layer_names(), self)





class GoodfellowNormalInvariance(NumpyMeasure):
    g_key="global"
    l_key="local"


    def __init__(self, alpha=0.99, sign=1):
        assert sign in [1,-1]
        super().__init__()
        self.alpha=alpha
        self.sign=sign

    def eval(self,activations_iterator:ActivationsIterator,verbose=False):
        self.g = GoodfellowNormalGlobalInvariance(self.alpha, self.sign)
        g_result = self.g.eval(activations_iterator,verbose)
        thresholds = g_result.extra_values[GoodfellowNormalGlobalInvariance.thresholds_key]
        self.l = GoodfellowNormalLocalInvariance(thresholds, self.sign)
        l_result = self.l.eval(activations_iterator,verbose)

        ratio = divide_activations(l_result.layers,g_result.layers)
        extra = {self.g_key:g_result,self.l_key:l_result}

        return MeasureResult(ratio, activations_iterator.layer_names(), self,extra_values=extra)


    def __repr__(self):
        return f"{self.__class__.__name__}(gp={self.alpha})"

    def name(self):
        return "Goodfellow Normal Invariance"

    def abbreviation(self):
        return "GFN"


