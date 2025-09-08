import abc

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from tmeasures.transformation import Transformation

from .transformations import PyTorchTransformation, PyTorchTransformationSet


class Dataset2D(Dataset):

    @property
    @abc.abstractmethod
    def len0(self)->int:
        pass

    @property
    @abc.abstractmethod
    def len1(self)->int:
        pass

    @property
    @abc.abstractmethod
    def T(self):
        pass

    def __len__(self):
        return self.len0 * self.len1

    def d1tod2(self,idx:int):
        return idx // self.len1, idx % self.len1

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = self.d1tod2(idx)
        if len(idx) != 2:
            raise ValueError(f"Invalid index: {idx}")
        i, j = idx
        return self.getitem2d(i, j)

    @abc.abstractmethod
    def getitem2d(self, i, j):
        pass

    def row_dataset(self, row: int):
        return RowDataset(self,row)

    @abc.abstractmethod
    def get_transformations(self, rows:list[int],cols:list[int])->list[PyTorchTransformation]:
        pass



class STDataset(Dataset2D):

    def __init__(self, dataset: Dataset, transformations: PyTorchTransformationSet,device=torch.device("cpu")):
        """
        @param dataset: Non iterable dataset from which to draw samples
        @param transformations: set of transformations to apply to samples
        """

        if isinstance(dataset, IterableDataset):
            raise ValueError(
                f"{IterableDataset} not supported; must specify a map-style dataset (https://pytorch.org/docs/stable/data.html#dataset-types)")
        self.dataset = dataset
        self.transformations = transformations
        self.device=device

    def len_dataset(self):
        return len(self.dataset)

    def len_transformations(self):
        return len(self.transformations)


class SampleTransformationDataset(STDataset):

    def T(self):
        return TransformationSampleDataset(self.dataset, self.transformations)

    def getitem2d(self, i, j):
        s = self.dataset[i].to(self.device)
        t = self.transformations[j]
        return t(s)

    @property
    def len0(self):
        return self.len_dataset()

    @property
    def len1(self):
        return self.len_transformations()

    def get_transformations(self, rows:list[int],cols:list[int])->list[PyTorchTransformation]:
        return [self.transformations[c] for c in cols]


class TransformationSampleDataset(STDataset):

    def getitem2d(self, i, j):
        t = self.transformations[i]
        s = self.dataset[j].to(self.device)
        return t(s)

    def T(self):
        return SampleTransformationDataset(self.dataset, self.transformations)

    @property
    def len1(self):
        return self.len_dataset()

    @property
    def len0(self):
        return self.len_transformations()

    def get_transformations(self, rows:list[int],cols:list[int])->list[PyTorchTransformation]:
        return [self.transformations[r] for r in rows]


class RowDataset(Dataset):

    def __init__(self, d: Dataset2D, row: int):
        self.d = d
        self.row = row

    def __getitem__(self, item):
        return self.d.getitem2d(self.row, item)

    def __len__(self):
        return self.d.len1



