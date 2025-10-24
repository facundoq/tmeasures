from pathlib import Path
from typing import Type, TypeVar
import torchvision
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import tempfile
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