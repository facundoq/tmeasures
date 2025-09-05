import torch
import pytest
import tmeasures as tm
import numpy as np
from numpy.testing import assert_allclose


class ConstantModel(torch.nn.Module):
    def __init__(self,value=torch.Tensor(0)) -> None:
        super().__init__()
        self.value=value
    def forward(self,x:torch.Tensor):
        n = x.shape[0]
        result =  self.value.expand(n,*self.value.shape)
        return result
    
class IdentityModel(torch.nn.Module):
    def __init__(self,) -> None:
        super().__init__()
    def forward(self,x:torch.Tensor):
        return x
    
class RandomModel(torch.nn.Module):
    def __init__(self,shape:tuple,mean=0.0,std=1.0) -> None:
        super().__init__()
        self.mean=mean
        self.std=std
        self.shape=shape
    def forward(self,x:torch.Tensor):
        n = x.shape[0]
        shape = (n,*self.shape)
        return torch.normal(mean=self.mean,std=self.std,size=shape)
        


class ConstantDataset(torch.utils.data.Dataset):
    def __init__(self,value=0,shape=(10,10)):
        super().__init__()
        self.dataset = torch.utils.data.TensorDataset(torch.ones(shape)*value)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        return self.dataset[index][0]
    
default_options = tm.pytorch.PyTorchMeasureOptions(batch_size=1024)
large_options = tm.pytorch.PyTorchMeasureOptions(batch_size=2**14,num_workers=12)

def assert_instance(measure,dataset,transformations,activations_model,expected_result,atol=1e-5,options=default_options):
    result = measure.eval(dataset,transformations,activations_model,options)
    
    result = result.numpy()
    for name,layer,expected_layer in zip(result.layer_names,result.layers,expected_result):
        assert_allclose(layer,expected_layer,err_msg=f"Error in {measure} for activation '{name}'",atol=atol)

        
def test_constant_model_invariance():
    output_shape = (2,2)
    output = torch.rand(output_shape)
    expected_results = np.zeros(output.shape)
    expected_results_normalized = np.ones(output.shape)
    model = torch.nn.Sequential(ConstantModel(output))
    measures_results = [
                    #   (tm.pytorch.SampleVarianceInvariance(),[expected_results]),
                      (tm.pytorch.TransformationVarianceInvariance(),[expected_results]),
                    #   (tm.pytorch.NormalizedVarianceInvariance(),[expected_results_normalized]),
                      ]
    n = 5
    transformations = RepeatedIdentitySet(n)
    default_options.batch_size=3
    dataset = ConstantDataset(2,(n,5))
    activations_model = tm.pytorch.AutoActivationsModule(model)
    for measure,expected_result in measures_results:
       assert_instance(measure,dataset,transformations,activations_model,expected_result,options=default_options)

class RepeatedIdentitySet(tm.pytorch.transformations.PyTorchTransformationSet):
    def __init__(self,transformations=1):
        super().__init__([tm.pytorch.transformations.IdentityTransformation()]*transformations)
    def valid_input(self):
        return True
    def copy(self):
        return self
    def id(self):
        return "Identity"

def atest_random_model_invariance():
    output_shape = (2,2)
    mean,std=2.0,3
    model = torch.nn.Sequential(RandomModel(output_shape,2,3))
    expected_results = np.ones(output_shape)*std
    expected_results_normalized = np.ones(output_shape)
    measures_results = [(tm.pytorch.SampleVarianceInvariance(),[expected_results]),
                      (tm.pytorch.TransformationVarianceInvariance(),[expected_results]),
                      (tm.pytorch.NormalizedVarianceInvariance(),[expected_results_normalized]),
                      ]
    sample_size_order = 8
    n = 10**sample_size_order
    atol = 10**(-np.sqrt(sample_size_order//2))
    transformations = RepeatedIdentitySet(n)
    dataset = ConstantDataset(2,(n,2))
    activations_model = tm.pytorch.AutoActivationsModule(model)
    large_options.batch_size = n
    for measure,expected_result in measures_results:
        assert_instance(measure,dataset,transformations,activations_model,expected_result,atol=1e-1,options=large_options)



if __name__ == "__main__":
    import logging
    logging.basicConfig()
    
    #set logger to info level
    tm.logger.setLevel(logging.INFO)

    test_constant_model_invariance()
    #atest_random_model_invariance()