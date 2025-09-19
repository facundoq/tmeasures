import torch
import pytest
import tmeasures as tm
import numpy as np
from numpy.testing import assert_allclose
from .utils import ConstantModel,ConstantDataset,RandomModel,RepeatedIdentitySet,MeasureFixture
from torch import nn

default_options = tm.pytorch.PyTorchMeasureOptions(batch_size=32)
large_options = tm.pytorch.PyTorchMeasureOptions(batch_size=2**14,num_workers=0)


def constant_model_invariance(n:int,bs:int):
    o = tm.pytorch.PyTorchMeasureOptions(batch_size=bs)
    transformations = RepeatedIdentitySet(n)
    dataset = ConstantDataset(n,torch.Tensor((1,)))
    transformations = RepeatedIdentitySet(n)
    output_shape = (2,2)
    output = torch.rand(output_shape)
    result = np.zeros(output.shape)
    result_nv = np.ones(output.shape)
    model = torch.nn.Sequential(ConstantModel(output))
    sv,tv,nv = measures
    return [
            MeasureFixture(model, sv,[result],dataset,transformations,options=o),
            MeasureFixture(model, tv,[result],dataset,transformations,options=o),
            MeasureFixture(model, nv,[result_nv],dataset,transformations,options=o),
    ]    
def constant_model_invariance_options():
    for n,bs in [(1,1),(5,3),(5,11),(100,20),(20,100)]:
        yield from constant_model_invariance(n,bs)
    
@pytest.mark.parametrize("f",constant_model_invariance_options())   
def test_constant_model_invariance(f:MeasureFixture):
    f.assert_fixture(atol=1e-5) 
        
measures = [tm.pytorch.SampleVarianceInvariance(),
            tm.pytorch.TransformationVarianceInvariance(),
            tm.pytorch.NormalizedVarianceInvariance()
            ]

def random_model_invariance_options():
    sample_size_order = 2
    n = 10**sample_size_order
    atol = 10**(-np.sqrt(sample_size_order//2))
    transformations = RepeatedIdentitySet(n)
    dataset = ConstantDataset(n,torch.Tensor((1,)))
    o = tm.pytorch.PyTorchMeasureOptions(batch_size=2**10,num_workers=0)
    large_options.batch_size = n
    output_shape = (2,2)
    mean,std=2.0,3
    model = torch.nn.Sequential(RandomModel(output_shape,2,3))
    result = np.ones(output_shape)*std
    result_nv = np.ones(output_shape)
    sv,tv,nv = measures
    return [
            MeasureFixture(model, sv,[result],dataset,transformations,options=o),
            MeasureFixture(model, tv,[result],dataset,transformations,options=o),
            MeasureFixture(model, nv,[result_nv],dataset,transformations,options=o),
        ]

@pytest.mark.parametrize("f",random_model_invariance_options())
def test_random_model_invariance(f:MeasureFixture):
    f.assert_fixture(atol=1e-1)

if __name__ == "__main__":
    import logging
    # logging.basicConfig()
    #set logger to info level
    # tm.logger.setLevel(logging.INFO)
    # test_random_model_invariance()