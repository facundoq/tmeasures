import torch
from torchvision import transforms
from torchvision.transforms import functional
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import transformation_measure as tm
from enum import Enum

class TransformationStrategy(Enum):
    random_sample="random_sample"
    iterate_all="iterate_all"


class ImageDataset(Dataset):

    def __init__(self, image_dataset, transformations:tm.TransformationSet=None, transformation_scheme:TransformationStrategy=None ):

        if transformation_scheme is None:
            transformation_scheme = TransformationStrategy.random_sample
        self.transformation_strategy = transformation_scheme

        self.dataset=image_dataset

        if transformations is None:
            self.transformations=[tm.IdentityTransformation()]
        else:
            self.transformations=list(transformations)
        self.n_transformations=len(self.transformations)
        self.n_samples=len(self.dataset)
        self.setup_transformation_pipeline()


    def setup_transformation_pipeline(self,):
        x, y = self.dataset.get_all()
        n, c, w, h = x.shape
        self.h = h
        self.w = w
        self.c=c
        self.mu, self.std = self.calculate_mu_std(x)

        # transformations = [transforms.ToPILImage(), ]
        #
        # transformations.append(transforms.ToTensor())
        # transformations.append(transforms.Normalize(mu, std))
        # return transforms.Compose(transformations)

    def calculate_mu_std(self,x:torch.Tensor):

        xf = x.float()
        dims = (0, 2, 3)

        mu = xf.mean(dim=dims,keepdim=True)
        std = xf.std(dim=dims,keepdim=True)

        std[std == 0] = 1
        return mu,std

    def __len__(self):
        if self.transformation_strategy==TransformationStrategy.random_sample:
            return self.n_samples
        elif self.transformation_strategy==TransformationStrategy.iterate_all:
            return self.n_samples*self.n_transformations
        else:
            raise ValueError(self.transformation_strategy)

    def __getitem__(self, idx):
        assert(isinstance(idx,int))
        x,y=self.get_batch(idx)
        # print(y.shape)
        return x[0,],y


    def transform_batch(self,x,i_transformation):
        x = x.float()
        x = (x-self.mu)/self.std
        for i in range(x.shape[0]):
            sample=x[i:i+1,:]
            t = self.transformations[i_transformation[i]]
            x[i,:] = t(sample)
        return x

    def get_batch(self,idx):
        if isinstance(idx,int):
            idx = [idx]
        if self.transformation_strategy == TransformationStrategy.iterate_all:
            i_sample=[i % self.n_samples for i in idx]
            i_transformation= [i // self.n_samples for i in idx]
        elif self.transformation_strategy == TransformationStrategy.random_sample:
            i_sample = idx
            i_transformation = np.random.randint(0,self.n_transformations,size=(len(idx),))
        else:
            raise ValueError(self.transformation_strategy)

        x,y=self.dataset.get_batch(i_sample)
        x=self.transform_batch(x,i_transformation)
        y=y.type(dtype=torch.LongTensor)
        return x, y


    def get_all(self):
        ids = list(range(len(self)))
        return self.get_batch(ids)
