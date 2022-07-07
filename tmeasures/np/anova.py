from typing import List, Tuple
from .base import NumpyMeasure
from ..measure import ActivationsByLayer
from .. import MeasureResult
from .activations_iterator import ActivationsIterator
from .stats_running import RunningMeanWelford
import scipy.stats

class ANOVAInvariance(NumpyMeasure):
    # alpha = degree of confidence
    # Typically 0.90, 0.95, 0.99
    def __init__(self, alpha:float=0.99,bonferroni:bool=True):
        super().__init__()
        self.anova_f_measure=ANOVAFInvariance()
        assert(alpha>0)
        assert (alpha <1)
        self.alpha=alpha
        self.bonferroni=bonferroni

    def __repr__(self):
        return f"ANOVAI(α={self.alpha},b={self.bonferroni})"

    def eval(self, activations_iterator: ActivationsIterator,verbose=False) -> MeasureResult:
        f_result=self.anova_f_measure.eval(activations_iterator)
        d_b=f_result.extra_values["d_b"]
        d_w = f_result.extra_values["d_w"]
        n_layers=len(f_result.layer_names)
        layers=f_result.layers
        if self.bonferroni:
            # use adjusted alphas to calculate critical values
            critical_values = []
            for layer in layers:
                adjusted_alpha=self.alpha/layer.size
                critical_values.append(scipy.stats.f.ppf(adjusted_alpha,dfn=d_b,dfd=d_w))
        else:
            critical_values = [scipy.stats.f.ppf(self.alpha,dfn=d_b,dfd=d_w)]*n_layers

        f_critical=zip(layers,critical_values)
        layers=[ f>critical_value for (f,critical_value) in f_critical]
        return MeasureResult(layers,f_result.layer_names,self,f_result.extra_values)

    def name(self):
        return "ANOVA"
    def abbreviation(self):
        return self.name()

class ANOVAFInvariance(NumpyMeasure):
    def __init__(self):
        super().__init__()


    def __repr__(self):
        return f"{self.__class__.__name__}()"


    def eval(self,activations_iterator:ActivationsIterator,verbose=False)->MeasureResult:

        # calculate mean(X_t)
        u_t,n_t=self.eval_means_per_transformation(activations_iterator)
        n_layers=len(activations_iterator.layer_names())
        # Calculate mean(X)
        u=self.eval_global_means(u_t,n_layers)
        # Calculate  mean_t[ (mean(X_t)-mean(X))^2], that is Var( mean(X_t) ). Normalized with T-1
        ssb, d_b=self.eval_between_transformations_ssd(u_t,u,n_t)

        # Calculate t: (mean(X_t)-X)², that is Var(X_t). Normalized with N-T
        ssw, d_w=self.eval_within_transformations_ssd(activations_iterator, u_t)
        # calculate
        f_score=self.divide_per_layer(ssb,ssw)

        return MeasureResult(f_score, activations_iterator.layer_names(), self, extra_values={"d_b":d_b, "d_w":d_w})

    def divide_per_layer(self,a_per_layer:ActivationsByLayer,b_per_layer:ActivationsByLayer)->Tuple[ActivationsByLayer,int]:
        return [a/b for (a,b) in zip(a_per_layer,b_per_layer)]

    def eval_within_transformations_ssd(self,activations_iterator:ActivationsIterator,means_per_layer_and_transformation:List[ActivationsByLayer],)->ActivationsByLayer:
            n_layers = len(activations_iterator.layer_names())

            ssdw_per_layer = [0] * n_layers
            samples_per_transformation = []
            for means_per_layer,(transformation, transformation_activations) in zip(means_per_layer_and_transformation, activations_iterator.transformations_first()):
                # calculate the variance of all samples for this transformation
                n_samples = 0
                for x, batch_activations in transformation_activations:
                    n_samples += x.shape[0]
                    for j, layer_activations in enumerate(batch_activations):
                        for i in range(layer_activations.shape[0]):
                            d=(layer_activations[i,]-means_per_layer[j])**2
                            ssdw_per_layer[j]=ssdw_per_layer[j]+d
                samples_per_transformation.append(n_samples)
            # divide by degrees of freedom
            degrees_of_freedom = (samples_per_transformation[0] - 1) * len(samples_per_transformation)
            ssdw_per_layer = [s / degrees_of_freedom for s in ssdw_per_layer]
            return ssdw_per_layer,degrees_of_freedom

    def eval_between_transformations_ssd(self,means_per_layer_and_transformation:List[ActivationsByLayer],global_means:ActivationsByLayer,samples_per_transformation:List[int])->Tuple[ActivationsByLayer,int]:
        '''

        :param means_per_transformation: has len(transformations), each item has len(layers)
        :param global_means: has len(layers)
        :return:
        '''
        n_layers=len(global_means)
        n_transformations=len(means_per_layer_and_transformation)
        ssdb_per_layer=[0]*n_layers
        for  i,means_per_layer in enumerate(means_per_layer_and_transformation):
            n=samples_per_transformation[i]
            for j in range(n_layers):
                ssdb_per_layer[j] += n* ((means_per_layer[j]-global_means[j])**2)
        degrees_of_freedom=(n_transformations-1)
        for j in range(n_layers):
            ssdb_per_layer[j]/=degrees_of_freedom
        return ssdb_per_layer,degrees_of_freedom

    def eval_global_means(self, means_per_layer_and_transformation:List[ActivationsByLayer], n_layers:int)->ActivationsByLayer:
        '''

        :param means_per_layer_and_transformation:
        :param n_layers:
        :return: The global means for each layer, averaging out the transformations
        '''
        # n_transformations = len(means_per_layer_and_transformation)
        global_means_running = [RunningMeanWelford() for i in range(n_layers)]
        for transformation_means in means_per_layer_and_transformation:
            # means_per_layer  has the means for a given transformation
            for i,layer_means in enumerate(transformation_means):
                global_means_running[i].update(layer_means)

        return [rm.mean() for rm in global_means_running]




    def eval_means_per_transformation(self,activations_iterator:ActivationsIterator)->Tuple[List[ActivationsByLayer],List[int]]:
        '''
        For all activations, calculates the mean activation value for each transformation
        param activations_iterator
        return A list of mean activation values for each activation in each layer
        The list of samples per transformation
        '''
        n_layers = len(activations_iterator.layer_names())
        means_per_transformation = []
        samples_per_transformation=[]
        for transformation, transformation_activations in activations_iterator.transformations_first():
            samples_variances_running = [RunningMeanWelford() for i in range(n_layers)]
            # calculate the variance of all samples for this transformation
            n_samples=0
            for x, batch_activations in transformation_activations:
                n_samples+=x.shape[0]
                for j, layer_activations in enumerate(batch_activations):
                    for i in range(layer_activations.shape[0]):
                        samples_variances_running[j].update(layer_activations[i,])
            samples_per_transformation.append(n_samples)
            means_per_transformation.append([rm.mean() for rm in samples_variances_running])
        return means_per_transformation,samples_per_transformation


