from tests.pytorch.datasets import cifar10
import tmeasures as tm
import torchvision

import torch


import dataclasses
from typing import Callable

from tmeasures.measure import MeasureResult
from tmeasures.pytorch import NormalizedVarianceInvariance
from tmeasures.pytorch.transformations.affine import RotationTransformationSet, ScaleTransformationSet, TranslationTransformationSet


@dataclasses.dataclass
class Fixture:
    model:Callable[[torch.device,torch.utils.data.Dataset],torch.nn.Module]
    measure:tm.pytorch.PyTorchMeasure
    transformations:tuple[str,tm.pytorch.transformations.TransformationSet]
    dataset:torch.utils.data.Dataset


transformation_sets = {
    "rotation":RotationTransformationSet([0.0,.25,.5,.75,]),
    "scale":ScaleTransformationSet([(0.5,0.5),(0.5,1.5),(1.5,0.5),(1.5,1.5),(1.0,1.0)]),"translation":TranslationTransformationSet([(0.5,0),(0.0,0.5),(-0.5,0),(0.0,-0.5),(0.0,0.0)])
    }


def model_loader(model_function,**kwargs):
    def loader(device,dataset):
        model = model_function(**kwargs)
        model = model.to(device)
        model.eval()
        return model
    return loader

def evaluate_fixture_default(f:Fixture)->MeasureResult:
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
    return measure_result


def cifar10_resnet_variance_invariance_fixture():
    model = model_loader(torchvision.models.resnet18,weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    return Fixture(model, NormalizedVarianceInvariance(), transformation_sets["rotation"], cifar10())


