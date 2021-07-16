#!/usr/bin/env python
# coding: utf-8

# # Define a CNN model that implements ObservableLayersModule
import os
import transformational_measures as tm
import torch
from torch import nn
import pickle
from pathlib import Path
# Class for PyTorch models that return intermediate results
from transformational_measures.pytorch import ObservableLayersModule

# Utility class, same as PyTorch Sequential but returns intermediate layer values
from transformational_measures.pytorch import SequentialWithIntermediates

class Flatten(nn.Module):
    def forward(self, input:torch.Tensor):
        return input.view(input.size(0), -1)
    
# Model definition
class CNN(ObservableLayersModule):
    def __init__(self,shape):
        super(CNN, self).__init__()
        self.shape=shape
        c,h,w=shape
        h_flat,w_flat=h//4,w//4
        filters=32
        filters2=filters*2
        flat=h_flat*w_flat*filters2
        self.model=SequentialWithIntermediates(
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

    # required by ObservableLayersModule
    def forward_intermediates(self,x):
        return self.model.forward_intermediates(x)
    
    # required by ObservableLayersModule
    # Taken care by SequentialWithIntermediates
    def activation_names(self):
        return self.model.activation_names()


    


# # Usual train/fit test/evaluate methods

# In[11]:


import torch.nn.functional as F
# train and test as usual
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    batches=len(train_loader)
    log_interval_batches=int(batches*args.log_interval)
    #device=model.device()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval_batches == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# 
# # Train model for MNIST
# 

# In[ ]:


from torchvision import datasets, transforms
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR

# CONFIG
class Options:
  def __init__(self,lr=0.1,gamma=0.7,epochs=1,
               seed=0,batch_size=256,test_batch_size=512,
               log_interval=0.2,dry_run=False):
    self.lr=lr
    self.gamma=gamma
    self.epochs=epochs
    self.seed=seed
    self.batch_size=batch_size
    self.test_batch_size=test_batch_size
    self.log_interval=log_interval
    self.dry_run=dry_run

args=Options()
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")

results_path=Path("results")

# DATASET
base_preprocessing=[
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ]
# Rotation data augmentation
# CHANGE THIS VARIABLE from 0 to 180 to train with different intensities of data augmentation
# More data augmentation will cause the network to be more invariant
degree_range= 180 # train with random rotations from -degree_range to degree_range
train_augmentation=[transforms.RandomRotation(degree_range)]
train_transform=transforms.Compose(train_augmentation+base_preprocessing)
measure_transform=transforms.Compose(base_preprocessing)
path='~/.torchvision_datasets/mnist'
dataset1 = datasets.MNIST(path, train=True, download=True,
                    transform=train_transform)
dataset2 = datasets.MNIST(path, train=False,
                    transform=train_transform) 
train_loader = torch.utils.data.DataLoader(dataset1,args.batch_size)
test_loader = torch.utils.data.DataLoader(dataset2,args.test_batch_size)



# TRAIN
model_path=results_path/ "model.pickle"

if model_path.exists():
    model = torch.load(model_path)
else:
    model = CNN((1,28,28)).to(device)
    print(f"Training network with device: {device}")
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    torch.save(model,model_path)


# # Measure model's invariance  to rotations
# 
# 

# In[ ]:


# Iterate over images from MNIST without labels
class NoLabelsMNIST(datasets.MNIST):
  def __getitem__(self, index):
    x,y=super().__getitem__(index)
    return x
  def __len(self):
      return 400

dataset_nolabels = NoLabelsMNIST(path, train=False, download=True,
                    transform=measure_transform)

# Create a set of rotation transformations
from transformational_measures.transformations.parameters import UniformRotation
from transformational_measures.transformations.pytorch.affine import AffineGenerator
rotation_parameters=UniformRotation(n=4,angles=360)
transformations=AffineGenerator(r=rotation_parameters)

# Define an iterator over activations for pytorch
iterator = tm.NormalPytorchActivationsIterator(model, dataset_nolabels, transformations, 
                                                batch_size=128,num_workers=3,use_cuda=use_cuda)

# evaluate measure, with the iterator


da=tm.DistanceAggregation(normalize=True,keep_shape=True)
#mean_pnt = tm.AggregateTransformation(tm.AggregateFunction.mean)
mean_pnt = tm.AggregateTransformation(axis=(0,))
measures = [

    tm.ANOVAInvariance(),
    tm.GoodfellowNormalInvariance(),
    tm.NormalizedVarianceInvariance(),
    tm.NormalizedVarianceInvariance(pre_normalization_transformation=mean_pnt),
    tm.NormalizedDistanceInvariance(da),
    tm.NormalizedDistanceInvariance(da,pre_normalization_transformation=mean_pnt),
    tm.NormalizedVarianceSameEquivariance(),
    tm.NormalizedDistanceSameEquivariance(da),
]

for measure in measures:

    exp_id=f"rot{degree_range}_{measure}"
    result_filepath= results_path / f'{exp_id}_result.pickle'
    if os.path.exists(result_filepath):
        print(f"Measure {measure} already evaluated,skipping ")
        continue
    else:
        print(f"Evaluating measure {measure}...")

    # evaluate measure
    measure_result = measure.eval(iterator,verbose=True)

    # Save result

    with open(result_filepath, 'wb') as f:
        pickle.dump(measure_result, f)
    # Load result (optional, in case you don't want to run the above or your session died)
    with open(result_filepath, 'rb') as f:
        measure_result=pickle.load(f)

    from transformational_measures import visualization
    results=[measure_result]
    plot_filepath=results_path / f"{exp_id}_by_layers.png"
    visualization.plot_collapsing_layers_same_model(results, plot_filepath)
    heatmap_filepath=results_path / f"{exp_id}_heatmap.png"
    visualization.plot_heatmap(measure_result,heatmap_filepath)
    #


# In[ ]:




