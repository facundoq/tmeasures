import torch
import pytest
import tmeasures as tm
import numpy as np
from numpy.testing import assert_allclose
from .utils import ConstantModel,ConstantDataset,RandomModel,RepeatedIdentitySet

default_options = tm.pytorch.PyTorchMeasureOptions(batch_size=1024)
large_options = tm.pytorch.PyTorchMeasureOptions(batch_size=2**14,num_workers=0)

def assert_instance(measure,dataset,transformations,activations_model,expected_result,atol=1e-5,options=default_options):
    result = measure.eval(dataset,transformations,activations_model,options)
    
    result = result.numpy()
    for name,layer,expected_layer in zip(result.layer_names,result.layers,expected_result):
        assert_allclose(layer,expected_layer,err_msg=f"Error in {measure} for activation '{name}'",atol=atol)

def constant_model_invariance_options():
    output_shape = (2,2)
    output = torch.rand(output_shape)
    expected_results = np.zeros(output.shape)
    expected_results_normalized = np.ones(output.shape)
    model = torch.nn.Sequential(ConstantModel(output))
    return [
                       (model,tm.pytorch.SampleVarianceInvariance(),[expected_results]),
                      (model,tm.pytorch.TransformationVarianceInvariance(),[expected_results]),
                       (model,tm.pytorch.NormalizedVarianceInvariance(),[expected_results_normalized]),
                      ]     

@pytest.mark.parametrize("model,measure,expected_result",constant_model_invariance_options())   
def test_constant_model_invariance(model:torch.nn.Module,measure:tm.pytorch.PyTorchMeasure,expected_result:list[np.ndarray]):
    
    
    for n,bs in [(1,1),(5,3),(5,11),(100,20),(20,100)]:
        o = tm.pytorch.PyTorchMeasureOptions(batch_size=bs)
        transformations = RepeatedIdentitySet(n)
        dataset = ConstantDataset(n,torch.Tensor((1,)))
        activations_model = tm.pytorch.AutoActivationsModule(model)
        assert_instance(measure,dataset,transformations,activations_model,expected_result,options=o)




def random_model_invariance_options():
    output_shape = (2,2)
    mean,std=2.0,3
    model = torch.nn.Sequential(RandomModel(output_shape,2,3))
    expected_results = np.ones(output_shape)*std
    expected_results_normalized = np.ones(output_shape)
    return [
                    (model, tm.pytorch.SampleVarianceInvariance(),[expected_results]),
                      (model, tm.pytorch.TransformationVarianceInvariance(),[expected_results]),
                      (model, tm.pytorch.NormalizedVarianceInvariance(),[expected_results_normalized]),
        ]
    
@pytest.mark.parametrize("model,measure,expected_result",random_model_invariance_options())
def test_random_model_invariance(model,measure,expected_result):
    sample_size_order = 2
    n = 10**sample_size_order
    atol = 10**(-np.sqrt(sample_size_order//2))
    transformations = RepeatedIdentitySet(n)
    dataset = ConstantDataset(n,torch.Tensor((1,)))
    activations_model = tm.pytorch.AutoActivationsModule(model)
    o = tm.pytorch.PyTorchMeasureOptions(batch_size=2**10,num_workers=0)
    large_options.batch_size = n
    assert_instance(measure,dataset,transformations,activations_model,expected_result,atol=1e-1,options=o)

if __name__ == "__main__":
    import logging
    # logging.basicConfig()
    
    #set logger to info level
    # tm.logger.setLevel(logging.INFO)
    for p in constant_model_invariance_options():
        test_constant_model_invariance(*p)
    # test_random_model_invariance()