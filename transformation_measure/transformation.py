from __future__ import annotations
from typing import List,Tuple,Sized,Iterable,Iterator
import numpy as np
import torch
import abc

class Transformation:

    @abc.abstractmethod
    def __call__(self, x):
        pass


class TransformationSet(Sized, Iterable[Transformation]):

    @abc.abstractmethod
    def __iter__(self)->Iterator[Transformation]:
        pass

    def __len__(self):
        return len(list(self.__iter__()))

    @abc.abstractmethod
    def id(self):
        pass

    @abc.abstractmethod
    def valid_input(self,shape:Tuple[int, ])->bool:
        pass

    @abc.abstractmethod
    def copy(self)->TransformationSet:
        pass

class IdentityTransformation(Transformation):

    def __call__(self, x):
        return x


