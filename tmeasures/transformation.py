from __future__ import annotations
from typing import List,Tuple,Sized,Iterable,Iterator
import numpy as np
import torch
import abc

class Transformation:

    @abc.abstractmethod
    def __call__(self, x):
        pass

    def parameters(self):
        return np.array([])

class InvertibleTransformation:

    @abc.abstractmethod
    def inverse(self) -> InvertibleTransformation:
        pass

class TransformationSet(list,Sized, Iterable[Transformation]):

    def __init__(self,members):
        super().__init__(members)

    @abc.abstractmethod
    def id(self):
        pass

    @abc.abstractmethod
    def valid_input(self,shape:Tuple[int, ])->bool:
        pass

    @abc.abstractmethod
    def copy(self)->'TransformationSet':
        pass

    @abc.abstractmethod
    def parameter_range(self):
        pass


class IdentityTransformation(InvertibleTransformation):
    def __call__(self, x):
        return x

    def inverse(self) -> InvertibleTransformation:
        return self


