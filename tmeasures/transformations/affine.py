
from tmeasures.transformation import Transformation,TransformationSet
from typing import Tuple,Iterator
import itertools
import abc
import torch
TranslationParameter = Tuple[float, float]
ScaleParameter = Tuple[float, float]
RotationParameter = float


class AffineParameters:
    def __init__(self,r:RotationParameter=0,s:ScaleParameter=(1,1),t:TranslationParameter=(0,0)):
        self.r=r
        self.s=s
        self.t=t
    def __iter__(self):
        return iter((self.r,self.s,self.t))

    def inverse(self):
        r, s, t = self.r,self.s,self.t
        r=-r
        tx,ty=t
        t= (-tx,-ty)
        sx,sy=s
        s=(1/sx,1/sy)
        return AffineParameters(r,s,t)

    def as_tensor(self):
        r,(sx,sy),(tx,ty) = self.r,self.s,self.t
        return torch.tensor([r,sx,sy,tx,ty])

    def __repr__(self):
        r, s, t = self
        return f"AffineParameters(r={r},t={t},s={s})"

class AffineTransformation(Transformation):
    def __init__(self,ap:AffineParameters):
        self.ap=ap

    def __eq__(self, other):
        if self.__class__ == other.__class:
            return self.ap == other.parameters
        else:
            return False

    def __repr__(self):
        r, s, t = self.ap
        return f"AffineTransformation(r={r},s={s},t={t})"

    @abc.abstractmethod
    def inverse(self):
        pass
    def parameters(self):
        return self.ap.as_tensor()

def ifnone(x,v):
    if x is None:
        return v
    else:
        return x

from .parameters import NoRotation,NoScale,NoTranslation

class AffineGenerator(TransformationSet):
    def __init__(self, r:[RotationParameter]=None, s:[ScaleParameter]=None, t:[TranslationParameter]=None):
        r=ifnone(r,NoRotation())
        s=ifnone(s,NoScale())
        t=ifnone(t,NoTranslation())
        self.r,self.s,self.t=r,s,t
        parameters = itertools.product(r,s,t)
        parameters = [AffineParameters(r,s,t) for r,s,t in parameters]
        self.transformations = [self.make_transformation(p) for p in parameters]
        super().__init__(self.transformations)

    def valid_input(self,shape:Tuple[int, ]) -> bool:
        return len(shape) == 4

    @abc.abstractmethod
    def make_transformation(self, ap:AffineParameters)->AffineTransformation:
        pass

    def __repr__(self):
        return f"Affine(r={self.r},s={self.s},t={self.t})"

    # def __eq__(self, other):
    #     if isinstance(other,self.__class__):
    #         other:SimpleAffineTransformationGenerator = other
    #         return self.r == other.r and self.s==other.s and self.t==other.t

    def id(self):
        return str(self)

    def inverse(self)->[Transformation]:
        return [t.inverse() for t in self]

