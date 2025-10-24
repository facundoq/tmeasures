import itertools

import torch
import pytest
import torchvision
from tests.pytorch.datasets import cifar10
from tests.pytorch.fixtures import Fixture, model_loader, evaluate_fixture_default, transformation_sets

import tmeasures as tm
import numpy as np
from numpy.testing import assert_allclose

import numpy as np


from tmeasures import transformations
from .utils import ConstantModel,ConstantDataset,RandomModel,RepeatedIdentitySet

def options():
    datasets = [cifar10()]
    models = [
        model_loader(torchvision.models.resnet18,weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1),
    ]
    average_fm=tm.pytorch.AverageFeatureMaps()
    measures = [
        tm.pytorch.NormalizedVarianceInvariance(average_fm),
        tm.pytorch.NormalizedVarianceInvariance(),
    ]
    fixtures = [Fixture(*f) for f in itertools.product(models,measures,transformation_sets.items(),datasets)]
    return fixtures

    

@pytest.mark.parametrize("f",options())
def test_cifar10(f:Fixture,):
    evaluate_fixture_default(f)
    