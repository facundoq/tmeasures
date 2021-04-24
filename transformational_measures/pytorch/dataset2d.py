from torch.utils.data import Dataset,IterableDataset
import abc
from ..transformation import TransformationSet

class Dataset2D(Dataset):

    @property
    @abc.abstractmethod
    def len0(self):
        pass

    @property
    @abc.abstractmethod
    def len1(self):
        pass

    @property
    @abc.abstractmethod
    def T(self):
        pass

    def len(self):
        return self.len0 * self.len1

    def __getitem__(self, idx):
        if len(idx) == 1:
            i, j = idx // self.len1, idx % self.len1
            return self.getitem2d(i, j)
        elif len(idx) == 2:
            i, j = idx
            return self.getitem2d(i, j)
        else:
            raise ValueError(f"Invalid index: {idx}.")

    @abc.abstractmethod
    def getitem2d(self, i, j):
        pass


class STDataset(Dataset2D):

    def __init__(self,dataset:Dataset,transformations:TransformationSet):
        """
        @param dataset: Non iterable dataset from which to draw samples
        @param transformations: set of transformations to apply to samples
        """

        if isinstance(self.dataset,IterableDataset):
            raise ValueError(f"{IterableDataset} not supported; must specify a map-style dataset (https://pytorch.org/docs/stable/data.html#dataset-types)")
        self.dataset=dataset
        self.transformations = transformations

class SampleTransformationDataset(STDataset):


    def T(self):
        return TransformationSampleDataset(self.dataset,self.transformations)

    def getitem2d(self, i, j):
        s = self.dataset[i]
        t = self.transformations[j]
        return t(s)

    def len0(self):
        return len(self.dataset)

    def len1(self):
        return len(self.transformations)

class TransformationSampleDataset(STDataset):

    def getitem2d(self, i, j):
        t = self.transformations[i]
        s = self.dataset[j]
        return t(s)

    def T(self):
        return SampleTransformationDataset(self.dataset,self.transformations)

    def len0(self):
        return len(self.transformations)

    def len1(self):
        return len(self.dataset)