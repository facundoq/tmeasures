from tmeasures import MeasureResult
from ..activations_iterator import ActivationsIterator
from ..base import NumpyMeasure
import tmeasures as tm
from multiprocessing import Queue
from .multithreaded_layer_measure import LayerMeasure,PerLayerMeasure,ActivationsOrder
import numpy as np
from ..stats_running import RunningMeanAndVarianceWelford, RunningMean
from scipy.stats import norm

class GlobalVarianceNormal(LayerMeasure):

    def __init__(self, id:int, name:str, alpha:float, sign:int):
        super().__init__(id,name)
        self.alpha = alpha
        self.sign=sign

    def eval(self,q:Queue,inner_q:Queue):
        running_mean = RunningMeanAndVarianceWelford()
        for transformation in self.queue_as_generator(q):
            i=0
            for activations in self.queue_as_generator(inner_q):
                if self.sign != 1:
                    activations *= self.sign
                running_mean.update_all(activations)

        std=running_mean.std()
        mean = running_mean.mean()
        original_shape = mean.shape
        mean=mean.reshape(mean.size)
        std = std.reshape(std.size)
        # calculate the threshold values (approximately)
        thresholds=np.zeros(mean.size)
        for i,(mu,sigma) in enumerate(zip(mean,std)):
            t=norm.ppf(self.alpha,loc=mu,scale=sigma)
            thresholds[i]=t
        #thresholds = mean+2*std
        thresholds=thresholds.reshape(original_shape)
        # set g(i) equal to the activations_percentage
        g= np.zeros_like(thresholds) + (1-self.alpha)
        return g,thresholds

class LocalVarianceNormal(LayerMeasure):

    def __init__(self, id:int, name:str, threshold:float,sign:int):
        super().__init__(id,name)
        self.threshold = threshold
        self.sign=sign

    def eval(self,q:Queue,inner_q:Queue):
        running_mean = RunningMean()
        # activation_sum=0
        n=0
        for transformation in self.queue_as_generator(q):
            for activations in self.queue_as_generator(inner_q):
                if self.sign != 1:
                    activations *= self.sign
                activated = (activations > self.threshold) * 1.0
                running_mean.update_all(activated)

        return running_mean.mean()




class GlobalFiringRateNormalMeasure(PerLayerMeasure):
    def __init__(self,activation_percentage:float,sign:int):
        super().__init__(ActivationsOrder.TransformationsFirst)
        self.activation_percentage:float=activation_percentage
        self.sign:int=sign

    def __repr__(self):
        return f"G()"

    def generate_layer_measure(self, i:int, name:str) -> LayerMeasure:
        return GlobalVarianceNormal(i, name, self.activation_percentage, self.sign)

    def generate_result_from_layer_results(self, results_tresholds, names):
        results_tresholds, tresholds = zip(*results_tresholds)
        return MeasureResult(results_tresholds, names, self, extra_values={"thresholds":tresholds})

class LocalFiringRateNormalMeasure(PerLayerMeasure):
    def __init__(self,thresholds:[np.ndarray],sign:int):
        super().__init__(ActivationsOrder.SamplesFirst)
        self.sign=sign
        self.thresholds=thresholds

    def __repr__(self):
        return f"L()"

    def generate_layer_measure(self, i:int, name:str) -> LayerMeasure:
        return LocalVarianceNormal(i, name, self.thresholds[i], self.sign)

class GoodfellowNormalMeasure(NumpyMeasure):


    def __init__(self, alpha=0.99, sign=1):
        assert sign in [1,-1]
        super().__init__()
        self.alpha=alpha
        self.sign=sign

    def eval(self,activations_iterator:ActivationsIterator):
        self.g = GlobalFiringRateNormalMeasure(self.alpha, self.sign)
        g_result = self.g.eval(activations_iterator)
        self.thresholds = g_result.extra_values["thresholds"]
        self.l = LocalFiringRateNormalMeasure(self.thresholds, self.sign)
        l_result = self.l.eval(activations_iterator)

        ratio = tm.divide_activations(l_result.layers,g_result.layers)
        return MeasureResult(ratio, activations_iterator.layer_names(), self)


    def __repr__(self):
        return f"GoodfellowNormal(gp={self.alpha})"

    def name(self):
        return "Goodfellow Normal"
    def abbreviation(self):
        return "GFN"