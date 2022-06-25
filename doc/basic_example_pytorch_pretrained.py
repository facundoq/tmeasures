#!/usr/bin/env python
# coding: utf-8

## requires
## pip install sklearn torchvision

# # Define a CNN model that implements ObservableLayersModule
import os
import tmeasures as tm
import torch
from torchvision import transforms,datasets, models
from pathlib import Path
import matplotlib.pyplot as plt

if __name__ == '__main__':
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = Path("~/tm_example_pretrained/").expanduser()

    # MODEL
    
    model = models.resnet18(pretrained=True)
    # Measure model's invariance  to rotations
    model = model.to(device)
    
    # DATASET
    preprocessing_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
        
    # Iterate over images from CIFAR10 without labels
    class Dataset(datasets.CIFAR10):
        def __getitem__(self, index):
            x, y = super().__getitem__(index)
            return x
    dataset_nolabels = Dataset(data_path, train=False, download=True,
                             transform=preprocessing_transforms,)

    # Get a subset of the whole dataset; no need for a large number of samples
    # to calculate the invariance
    N = 1000
    import numpy as np
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Subset
    indices, _ = train_test_split(np.arange(len(dataset_nolabels)), train_size=N, stratify=dataset_nolabels.targets,random_state=0)
    dataset_nolabels = Subset(dataset_nolabels, indices)


    # Create a set of 128 rotation transformations, with angles from 0 to 360
    from tmeasures.transformations.parameters import UniformRotation
    from tmeasures.pytorch.transformations.affine import AffineGenerator

    rotation_parameters = UniformRotation(n=128, angles=1.0)
    transformations = AffineGenerator(r=rotation_parameters)


    # evaluate measure
    model.eval()
    activations_module = tm.pytorch.AutoActivationsModule(model)
    
    print("Activations in model:")
    print(activations_module.activation_names())
    
    # Define the measure
    average_fm=tm.pytorch.AverageFeatureMaps()
    measure = tm.pytorch.NormalizedVarianceInvariance(average_fm)
    measure = tm.pytorch.TransformationVarianceInvariance()
  
    print(f"Evaluating measure {measure}...")
    # evaluate measure

    options = tm.pytorch.PyTorchMeasureOptions(batch_size=16, num_workers=0,model_device=device,measure_device="cpu",data_device="cpu")
    measure_result:tm.pytorch.PyTorchMeasureResult = measure.eval(dataset_nolabels,transformations,activations_module,options)
    measure_result = measure_result.numpy()
    
    from tmeasures import visualization

    f = tm.visualization.plot_average_activations(measure_result)
    plt.savefig(data_path / f"average_by_layer.png")
    plt.close()

    f = tm.visualization.plot_heatmap(measure_result)
    plt.savefig(data_path / "heatmap.png")
    plt.close()
    


