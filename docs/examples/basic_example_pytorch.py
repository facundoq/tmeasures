#!/usr/bin/env python
# coding: utf-8

## requires
## pip install tqdm poutyne sklearn

# # Define a CNN model that implements ObservableLayersModule
import os
import tmeasures as tm
import torch
import matplotlib.pyplot as plt
from torch import nn

from tmeasures.pytorch.model import AutoActivationsModule, ManualActivationsModule

class Flatten(nn.Module):
    def forward(self, input: torch.Tensor):
        return input.view(input.size(0), -1)


# Model definition
class CNN(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        c, h, w = shape
        h_flat, w_flat = h // 4, w // 4
        filters = 32
        filters2 = filters * 2
        flat = h_flat * w_flat * filters2
        self.model = nn.Sequential(
            nn.Conv2d(c, filters, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(filters, filters2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            Flatten(),
            nn.Linear(flat, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=-1),
        ) 

    # forward works as normal
    def forward(self, x):
        return self.model.forward(x)


from pathlib import Path
from torchvision import datasets, transforms
from poutyne import Model

import pickle

if __name__ == '__main__':


    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_path = Path("~/.tm_example_pytorch/").expanduser()
    results_path.mkdir(parents=True, exist_ok=True)

    # DATASET
    base_preprocessing = [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]
    # Rotation data augmentation
    # CHANGE THIS VARIABLE from 0 to 180 to train with different intensities of data augmentation
    # More data augmentation will cause the network to be more invariant
    degree_range = 180  # train with random rotations from -degree_range to degree_range
    train_augmentation = [transforms.RandomRotation(degree_range)]
    train_transform = transforms.Compose(train_augmentation + base_preprocessing)
    measure_transform = transforms.Compose(base_preprocessing)
    path = results_path / 'mnist'

    train_dataset = datasets.MNIST(path, train=True, download=True,
                                   transform=train_transform)
    test_dataset = datasets.MNIST(path, train=False,
                                  transform=train_transform)


    # TRAIN
    model_path = results_path / "model.pickle"

    if model_path.exists():
        model = torch.load(model_path)
    else:
        model = CNN((1, 28, 28))
        poutyne_model = Model(model,
                              optimizer='adam',
                              loss_function='cross_entropy',
                              batch_metrics=['accuracy'],
                              device=device)
        poutyne_model.fit_dataset(train_dataset, test_dataset, epochs=5, batch_size=128,num_workers=2  ,dataloader_kwargs={"pin_memory":True})
        torch.save(model, model_path)


    # Measure model's invariance  to rotations

    # Iterate over images from MNIST without labels
    # Using same name as before to avoid double download
    class MNIST(datasets.MNIST):
        def __getitem__(self, index):
            x, y = super().__getitem__(index)
            return x
    dataset_nolabels = MNIST(path, train=False, download=True,
                             transform=measure_transform,)

    import numpy as np
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Subset
    indices, _ = train_test_split(np.arange(len(dataset_nolabels)), train_size=1000, stratify=dataset_nolabels.targets,random_state=0)
    dataset_nolabels = Subset(dataset_nolabels, indices)


    # Create a set of rotation transformations
    from tmeasures.transformations.parameters import UniformRotation
    from tmeasures.pytorch.transformations.affine import AffineTransformationSet

    rotation_parameters = UniformRotation(n=128, angles=1.0)
    transformations = AffineTransformationSet(r=rotation_parameters)



    model.eval()
    activations = tm.pytorch.get_activations(model)
    # remove activations that cant be inverted for SameEquivariance
    for i in range(6):
        activations.pop(next(iter(activations)))
                        
    print("Activations in model:")
    print(activations.keys())
    filtered_model = ManualActivationsModule(model,activations)
    activations_module = AutoActivationsModule(model)
    
    average_fm=tm.pytorch.AverageFeatureMaps()
    measures = [
        (tm.pytorch.TransformationVarianceInvariance(),),
        # (tm.pytorch.SampleVarianceInvariance(),activations_module),
        # (tm.pytorch.NormalizedVarianceInvariance(),activations_module),
        # (tm.pytorch.NormalizedVarianceInvariance(average_fm), activations_module),
        # (tm.pytorch.TransformationVarianceSameEquivariance(),filtered_model),
        # (tm.pytorch.SampleVarianceSameEquivariance(),filtered_model),
        # (tm.pytorch.NormalizedVarianceSameEquivariance(),filtered_model),
        # (tm.pytorch.NormalizedVarianceSameEquivariance(average_fm),filtered_model),
    ]

    for measure,model in measures:
        exp_id = f"rot{degree_range}_{measure}"
        result_filepath = results_path / f'{exp_id}_result.pickle'
        if os.path.exists(result_filepath):
            print(f"Measure {measure} already evaluated, loading...")
            # Load result (optional, in case you don't want to run the above or your session died)
            with open(result_filepath, 'rb') as f:
                measure_result = pickle.load(f)
        else:
            print(f"Evaluating measure {measure}...")
            # evaluate measure

            options = tm.pytorch.PyTorchMeasureOptions(batch_size=128, num_workers=0,model_device=device,measure_device=device,data_device="cpu")
            measure_result:tm.pytorch.PyTorchMeasureResult = measure.eval(dataset_nolabels,transformations,model,options)
            measure_result = measure_result.numpy()
            # Save result

            with open(result_filepath, 'wb') as f:
                pickle.dump(measure_result, f)

        f = tm.visualization.plot_average_activations(measure_result)
        f.tight_layout()
        plt.savefig(results_path / f"{exp_id}_by_layers.png")
        plt.close()

        f = tm.visualization.plot_heatmap(measure_result)
        f.tight_layout()
        plt.savefig(results_path / f"{exp_id}_heatmap.png")
        plt.close()

