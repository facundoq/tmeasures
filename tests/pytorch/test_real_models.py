import itertools
from pathlib import Path
import tempfile
from typing import Callable, Type, TypeVar
import torch
import pytest
import torchvision
import tmeasures as tm
import numpy as np
from numpy.testing import assert_allclose

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from tmeasures import transformations
from .utils import ConstantModel,ConstantDataset,RandomModel,RepeatedIdentitySet
from tmeasures.pytorch.transformations.affine import RotationTransformationSet,ScaleTransformationSet,TranslationTransformationSet

transformation_sets = {
    "rotation":RotationTransformationSet([0.0,.25,.5,.75,]),    
    "scale":ScaleTransformationSet([(0.5,0.5),(0.5,1.5),(1.5,0.5),(1.5,1.5),(1.0,1.0)]),"translation":TranslationTransformationSet([(0.5,0),(0.0,0.5),(-0.5,0),(0.0,-0.5),(0.0,0.0)])
    }

import dataclasses

@dataclasses.dataclass
class Fixture:
    model:Callable[[torch.device,torch.utils.data.Dataset],torch.nn.Module]
    measure:tm.pytorch.PyTorchMeasure
    transformations:tuple[str,tm.pytorch.transformations.TransformationSet]
    dataset:torch.utils.data.Dataset


def model_loader(model_function,**kwargs):
    def loader(device,dataset):
        model = model_function(**kwargs)
        model = model.to(device)
        model.eval()
        return model
    return loader


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

T = TypeVar('T', bound=torch.utils.data.Dataset)

def dataset_for_tmeasures(dataset_class:Type[T],mean,std,N=20):
    tmp_path = tempfile.gettempdir()
    preprocessing_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean,
                                 std=std)
    ])
    data_path = Path(tmp_path).expanduser()
    # Iterate over images from CIFAR10 without labels
    class Dataset(dataset_class,torchvision.datasets.VisionDataset):
        def __getitem__(self, index):
            x, y = super().__getitem__(index)
            return x
    dataset_nolabels = Dataset(data_path, train=False, download=True,
                             transform=preprocessing_transforms,)
    # Get a subset of the whole dataset; no need for a large number of samples
    # to calculate the invariance
    indices, _ = train_test_split(np.arange(len(dataset_nolabels)), train_size=N, stratify=dataset_nolabels.targets,random_state=0)
    dataset_nolabels = Subset(dataset_nolabels, indices)
    return dataset_nolabels

def cifar10(N=20):
    return dataset_for_tmeasures(torchvision.datasets.CIFAR10,
                                 mean = [0.4914, 0.4822, 0.4465],
                                 std=[0.2470, 0.2435, 0.2616],
                                 N=N
                                 )
def mnist(N=20):
    class RGBMNIST(torchvision.datasets.MNIST):
        def __getitem__(self, index):
            x, y = super().__getitem__(index)
            x = x.repeat(3,1,1)
            return x,y
    return dataset_for_tmeasures(RGBMNIST,
                                 mean = [0.1307,],
                                 std=[0.3081],
                                 N=N
                                 )    
    

@pytest.mark.parametrize("f",options())
def test_cifar10(f:Fixture,):
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu_device =torch.device("cpu")
    t_name,t_set = f.transformations
    dataset = f.dataset
    model = f.model(device,dataset)

    activations_module = tm.pytorch.AutoActivationsModule(model)
    print("Activations in model:")
    print(activations_module.activation_names())
  
    print(f"Evaluating measure {f.measure} with model {model} and transformations {t_name}...")
    # evaluate measure

    options = tm.pytorch.PyTorchMeasureOptions(batch_size=16, num_workers=0,model_device=device,measure_device=device,data_device=cpu_device)
    measure_result:tm.pytorch.PyTorchMeasureResult = f.measure.eval(dataset,t_set,activations_module,options)
    measure_result = measure_result.numpy()
    