#%%
from tempfile import TemporaryDirectory
import numpy as np
import matplotlib.pyplot as plt

#%pip install tmeasures 
#%pip install statsmodels

#%pip install tinyimagenet
#%pip install scikit-learn
import tmeasures as tm
import logging
tm.logger.setLevel(logging.INFO)

import torch 
from torchvision import models
from pathlib import Path

### para mostrar la trasnformacion
import tinyimagenet
import torchvision
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results_path = Path(TemporaryDirectory().name)/"tm_example_pytorch"
results_path.mkdir(parents=True, exist_ok=True)
print(f"Saving results to {results_path}")
# %%

model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
# model = models.vit_b_16(weights = models.ViT_B_16_Weights.IMAGENET1K_V1)
model = model.to(device)


# Put the model in evaluation mode
model.eval()
# print(model)


#%%


class TinyImageNet(tinyimagenet.TinyImageNet):
    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return x

normalize_transform = torchvision.transforms.Compose(
    [
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize(TinyImageNet.mean,TinyImageNet.std),
     torchvision.transforms.Resize((224,224))
     ])


dataset_nolabels = TinyImageNet(root="~/.datasets/tinyimagenet/",split="test", transform=normalize_transform)
#%%
# Subsample 
N = 200
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
indices, _ = train_test_split(np.arange(len(dataset_nolabels)), train_size=N, stratify=dataset_nolabels.targets,random_state=0)
dataset_nolabels = Subset(dataset_nolabels, indices)
#%%
import math

def brightness_transform(brightness_factor:float):
    #return 
    # lambda x: torchvision.transforms.functional.adjust_brightness(x,brightness_factor)
    return lambda x: x*brightness_factor

transformations = [brightness_transform(factor) for factor in [0.25,0.5,0.75,1,1.25,1.50,1.75,2.0]]

def plot_transformed_images():
    n=500
    step = 50
    transformed_images = [ t(dataset_nolabels[i]) for i in range(0,n,step) for t in transformations]
    grid = torchvision.utils.make_grid(transformed_images)


    grid_np = grid.permute(1,2,0).numpy()
    grid_np = grid_np * np.array(TinyImageNet.std)+np.array(TinyImageNet.mean)
    plt.figure(dpi=200)
    plt.imshow(grid_np)
# plot_transformed_images()
#%%



# Create an ActivationsModule from the vanilla model

layer_count = 0

def filter_stochastic(a):
    if  str(a).startswith("StochasticDepth"):
        return False
    global layer_count
    if layer_count<5:
        layer_count+=1
        return True
    else:
        return False

activations = tm.pytorch.get_activations(model)
activations = {k:v for k,v in activations.items() if filter_stochastic(k)}
activations_module = tm.pytorch.ActivationsModule(model,activations)


    

#%%
# Define options for computing the measure
options = tm.pytorch.PyTorchMeasureOptions(batch_size=4, num_workers=0,model_device=device,measure_device=device,data_device="cpu")
#%%
# Define the measure and evaluate it
#measure = tm.pytorch.NormalizedVarianceInvariance()
measure = tm.pytorch.SampleVarianceInvariance()
measure_result:tm.pytorch.PyTorchMeasureResult = measure.eval(dataset_nolabels,transformations,activations_module,options)  
measure_result = measure_result.numpy()
# %%
vec_inv = tm.pytorch.PyTorchMeasureResult.per_layer_average(measure_result)
print(measure_result.layer_names)
print(vec_inv)

