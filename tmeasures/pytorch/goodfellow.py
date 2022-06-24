from math import floor
from typing import Dict, List
import tmeasures as tm
from torch.utils.data import Dataset

from tmeasures.pytorch.transformations import IdentityTransformationSet

from .stats_running import RunningMeanAndVarianceWelford, RunningMeanWelford
from .base import PyTorchLayerMeasure, PyTorchMeasure, PyTorchMeasureByLayer, PyTorchMeasureOptions, PyTorchMeasureResult, STMatrixIterator
from .activations_iterator import PytorchActivationsIterator
from . import ActivationsModule
from .layer_measures import Variance
from .quotient import  divide_activations
import torch


default_alpha=0.99
default_sign=1


class PercentActivationThreshold(PyTorchLayerMeasure):
    def __init__(self,sign:float=1,percent:torch.DoubleTensor=0.99) -> None:
        super().__init__()
        self.percent=percent
        self.sign=sign
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sign={self.sign},p={self.percent})"

    def eval(self, st_iterator: STMatrixIterator, layer_name: str):
        n = len(st_iterator)
        values = torch.Tensor()
        for row, row_iterator in enumerate(st_iterator):
            if row == 0:
                m = len(row_iterator)
            for batch_i, batch_activations in enumerate(row_iterator):
                if row == 0 and batch_i == 0:
                    # initialize matrix to store activation values 
                    # One row for each activation
                    values = torch.zeros((n,m,*batch_activations.shape))
                    
                values[row,batch_i,:] = batch_activations*self.sign
        # change shape of values to Activations x Samples
        # where each row has all the samples of a particular activation
        original_shape = values.shape[3:]
        print("before",values.shape)
        samples_n = values.shape[0]*values.shape[1]*values.shape[2]
        values = values.reshape((samples_n,-1)).transpose(1,0)
        # sort rows, one for each activation
        # print(values.shape)
        values, _ = torch.sort(values)
        # determine index of threshold
        i = floor(values.shape[1]*self.percent)
        thresholds= values[:,i]
        # reshape thresholds to original shape
        thresholds = thresholds.reshape(original_shape)
        return thresholds

class NormalPValueThreshold(PyTorchLayerMeasure):
    def __init__(self,sign:float=1,alpha:torch.DoubleTensor=0.99) -> None:
        super().__init__()
        self.alpha=alpha
        self.sign=sign

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sign={self.sign},alpha={self.alpha})"

    def eval(self, st_iterator: STMatrixIterator, layer_name: str):
        
        mean = RunningMeanAndVarianceWelford()
        for row, row_iterator in enumerate(st_iterator):
            for batch_n, batch_activations in enumerate(row_iterator):
                mean.update_batch(batch_activations.double()*self.sign)

        
        # return mean.mean()
        μ,σ = mean.mean(),mean.std()
        original_shape = μ.shape
        
        σ[σ<1e-16]=1e-16
        d = torch.distributions.Normal(μ,σ)
        alpha = torch.tensor([self.alpha]).to(μ.device)
        p_values = d.icdf(alpha)
        
        # μ,σ = μ.reshape(μ.numel()),σ.reshape(σ.numel())
        # p_values=torch.zeros(μ.shape)
        # alpha = torch.tensor([self.alpha]).to(μ.device)
        # for j,(mu,sigma) in enumerate(zip(μ,σ)):
        #     if sigma>0:
        #         d = torch.distributions.Normal(mu,sigma)
        #         t = d.icdf(alpha)
        #     else:
        #         t=mu
        #     p_values[j]=t
        # print(f"NormalPvalue {layer_name} p_values: {p_values.shape} ")
        # p_values = p_values.reshape(original_shape)
        
        return p_values
   
class MeanFiringRate(PyTorchLayerMeasure):
    def __init__(self,sign:float,thresholds:Dict[str,torch.Tensor]) -> None:
        super().__init__()
        self.sign=sign
        self.thresholds=thresholds

    def eval(self, st_iterator: STMatrixIterator, layer_name: str):
        thresholds=self.thresholds[layer_name]
        mean = RunningMeanWelford()
        for row, row_iterator in enumerate(st_iterator):
            for batch_n, batch_activations in enumerate(row_iterator):
                if self.sign != 1:
                    batch_activations *= self.sign
                values = (batch_activations > thresholds) * 1.0
                mean.update_all(values)
        return mean.mean()


class GoodfellowInvariance(PyTorchMeasure):

    g_key="global"
    l_key="local"
    t_key= "thresholds"

    def __init__(self, sign=1, global_transformations = IdentityTransformationSet(),threshold_algorithm=NormalPValueThreshold(sign=1,alpha=0.99)):
        assert sign in [1,-1]
        super().__init__()
        self.sign=sign
        self.threshold_algorithm=threshold_algorithm
        self.global_transformations = global_transformations
        

    def eval(self, dataset: Dataset, transformations: tm.TransformationSet, model: ActivationsModule,o: PyTorchMeasureOptions):
        
        threshold_measures = PyTorchMeasureByLayer(self.threshold_algorithm)
        
        thresholds = threshold_measures.eval(dataset,self.global_transformations,model,o)
        
        mean_firing_rate = MeanFiringRate(self.sign,thresholds.layers_dict())
        
        g_result = PyTorchMeasureByLayer(mean_firing_rate).eval(dataset,self.global_transformations,model,o)
        
        l_result = PyTorchMeasureByLayer(mean_firing_rate).eval(dataset,transformations,model,o)
        
        # self.g = GoodfellowNormalGlobalInvariance(self.alpha, self.sign)
        # g_result = self.g.eval(dataset, transformations, model,o)
        # thresholds = g_result.extra_values[GoodfellowNormalGlobalInvariance.thresholds_key]
        # self.l = GoodfellowNormalLocalInvariance(thresholds, self.sign)
        # l_result = self.l.eval(dataset, transformations, model,o)

        ratio = divide_activations(l_result.layers,g_result.layers)
        extra = {self.g_key:g_result,self.l_key:l_result,self.t_key:thresholds}

        return PyTorchMeasureResult(ratio, model.activation_names(), self,extra_values=extra)

    def __repr__(self):
        return f"{self.abbreviation()}({self.threshold_algorithm})"
