
"""# Measure invariance in a pretrained EfficientNet model
#Using PyTorch, we will download a EfficientNet_b0 model pretrained on ImageNet and evaluate its invariance."""


# Commented out IPython magic to ensure Python compatibility.
# %pip install tmeasures 
# %pip install tinyimagenet
# %pip install scikit-learn

# %load_ext autoreload
# %autoreload 2
import torch 

from pathlib import Path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results_path = Path("~/tm_example_pytorch/").expanduser()
results_path.mkdir(parents=True, exist_ok=True)

from torchvision import models


# model = models.efficientnet_b0(weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1)
model = models.efficientnet_v2_s()
model = model.to(device)

"""# Measure model's invariance 
To measure the model's invariance, we require three things:

A version of the dataset, without labels and reduced in size to reduce computation.
A discrete and finite set of transformations
The model itself, with access to intermediate values or activations
Afterwards, we can create an Invariance Measure and compute it with these elements.

# 1. Dataset
Since the invariance measure do not use the labels of the dataset, we will create a custom TinyImageNet dataset which does not return labels, only samples.

Also, since the invariance measure does not require large sample sizes, we will subsample the test set of mnist to obtain a reduced sample and reduce computation time.
"""

import tinyimagenet
import torchvision

class TinyImageNet(tinyimagenet.TinyImageNet):
    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return x

normalize_transform = torchvision.transforms.Compose(
    [
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize(TinyImageNet.mean,TinyImageNet.std),
     ])


dataset_nolabels = TinyImageNet(root="~/.datasets/tinyimagenet/",split="test", transform=normalize_transform)


# Subsample 
N = 1000
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
indices, _ = train_test_split(np.arange(len(dataset_nolabels)), train_size=N, stratify=dataset_nolabels.targets,random_state=0)
dataset_nolabels = Subset(dataset_nolabels, indices)

"""# 2. Model
PyTorch works with nn.Module objects that provide a forward method. The invariance measures, however, do not require just the result(s) of the forward method. Instead, we need the result of all the intermediate values or activations used to compute the final output(s).

While it would be possible to modify a nn.Module defined model to return all of its activations, this would be cumbersome and difficult to manage since when training/testing we would require the usual forward, and when computing the measure we would require the new forward.

Therefore, measures in tmeasures require a model that implements the ActivationsModule interface with just two methods: forward_activations and activation_names. While implementing these methods allow you to best decide which activations are selected and how they are used, they can be cumbersome to define and mantain, specially when actively modifying a model.

Luckily, the AutoActivationsModule can take an unmodified nn.Module and automatically implement these methods using [forward hooks](https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_hook.html).
"""

import tmeasures as tm


# Put the model in evaluation mode
model.eval()
print(model)


# Create an ActivationsModule from the vanilla model
def filter_stochastic(a):
    return not str(a).startswith("StochasticDepth")

activations_module = tm.pytorch.AutoActivationsModule(model,filter=filter_stochastic)
print(len(activations_module.names),len(activations_module.activations))
model.eval()
for a in activations_module.activations:
    print(a,a.training)
    # a.training=False
"""# Computing the measure
Last step before computing the measure: we need to define a PyTorchMeasureOptions object to configure where and the measure will be computed. The batch_size and num_workers keywords are analogous to the ones used in PyTorch's DataLoader.

The data_device, model_device and measure_device indicate, respectively, where the transformations and data preprocessing is performed, where the activations of the model are computed, and finally where the actual measure is computed. In simple cases, these devices could all be the same.

Finally, we can eval the measure with the dataset, transformation, model and options, obtaining a PyTorchMeasureResult, which can be handily converted to a numpy version for easy visualization.
"""

import torchvision
import math

def brightness_transform(brightness_factor:float):
    #return lambda x: torchvision.transforms.functional.adjust_brightness(x,brightness_factor)
    return lambda x: x*brightness_factor

transformations = [brightness_transform(factor) for factor in [0.25,0.5,0.75,1,1.25,1.50,1.75,2.0]]

n=500
step = 50
transformed_images = [ t(dataset_nolabels[i]) for i in range(0,n,step) for t in transformations]
grid = torchvision.utils.make_grid(transformed_images)

import matplotlib.pyplot as plt
grid_np = grid.permute(1,2,0).numpy()
grid_np = grid_np * np.array(TinyImageNet.std)+np.array(TinyImageNet.mean)
plt.figure(dpi=200)
plt.imshow(grid_np)

# Define options for computing the measure
options = tm.pytorch.PyTorchMeasureOptions(batch_size=128, num_workers=0,model_device=device,measure_device=device,data_device="cpu")

# Define the measure and evaluate it
measure = tm.pytorch.NormalizedVarianceInvariance()
measure_result:tm.pytorch.PyTorchMeasureResult = measure.eval(dataset_nolabels,transformations,activations_module,options)  ## lista de varianzas de cada capa
measure_result = measure_result.numpy()

import matplotlib.pyplot as plt
tm.visualization.plot_average_activations(measure_result)
plt.show()
tm.visualization.plot_heatmap(measure_result)

