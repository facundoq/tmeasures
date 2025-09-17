import math
from typing import Iterator, List

import numpy as np
import torch
from torch.nn import functional as F

from tmeasures.pytorch.transformations import PyTorchTransformation, PyTorchTransformationSet
from tmeasures.transformations import affine
from tmeasures.transformations.affine import RotationParameter, ScaleParameter, TranslationParameter
from tmeasures.transformations.parameters import NoRotation, NoScale, NoTranslation


class AffineTransformation(affine.AffineTransformation,PyTorchTransformation):

    def __init__(self,ap:affine.AffineParameters):
        super().__init__(ap)
        self.transformation_matrix= self.get_transformation_matrix()
        self.transformation_matrix= self.transformation_matrix.unsqueeze(0)

    def get_transformation_matrix(self)->torch.Tensor:
        # center_matrix=torch.eye(3)
        # # center_matrix[:2,2]=-.5
        # decenter_matrix=torch.eye(3)
        # # decenter_matrix[:2,2]=.5
        p =self.ap
        r= p.r*180
        sx,sy=p.s
        #tx,ty=t
        matrix = torch.eye(3)
        matrix[:2, :2] = torch.tensor([[np.cos(r)/sx, -1.0*np.sin(r)/sy],
                                        [np.sin(r)/sx, np.cos(r)/sy]])
        matrix[:2,2]=torch.tensor(p.t)*2
        #matrix=decenter_matrix.mm(matrix.mm(center_matrix))
        # print(matrix)
        return matrix[:2,:]

    def __call__(self, x: torch.FloatTensor):
        with torch.no_grad(): # TODO is this necessary?

            c, h, w = x.shape
            x = torch.unsqueeze(x,0)
            shape=[1,c,h,w]
            grid = F.affine_grid(self.transformation_matrix,shape ,align_corners=False)
            grid=grid.to(x.device)
            x = F.grid_sample(x, grid,align_corners=False,padding_mode="border")
            return x[0,]

    def inverse(self):
        return AffineTransformation(self.ap.inverse())


class RotationTransformation(AffineTransformation):
    def __init__(self,p:affine.RotationParameter):
        self.p=p
        ap=affine.AffineParameters(r=p)
        super().__init__(ap)

    def parameters(self):
        # encode with cos and sin
        # https://arxiv.org/pdf/1702.01499.pdf0
        return torch.tensor([math.cos(self.p*360),math.sin(self.p*360)])

    def inverse(self):
        return RotationTransformation(-self.p)

class ScaleTransformation(AffineTransformation):
    def __init__(self,p:affine.ScaleParameter):
        self.p=p
        ap=affine.AffineParameters(s=p)
        super().__init__(ap)

    def parameters(self):
        sx,sy=self.p
        return torch.tensor([sx,sy])
    def inverse(self):
        sx,sy=self.p
        return ScaleTransformation((1/sx,1/sy))

class TranslationTransformation(AffineTransformation):
    def __init__(self,p:affine.TranslationParameter):
        self.p=p
        ap=affine.AffineParameters(t=p)
        super().__init__(ap)
    def parameters(self):
        tx, ty = self.p
        return torch.tensor([tx, ty])

    def inverse(self):
        tx,ty=self.p
        return TranslationTransformation((-tx,-ty))

class BaseAffineTransformationSet(affine.AffineGenerator,PyTorchTransformationSet):
    def __init__(self, r:list[RotationParameter]=NoRotation(), s:list[ScaleParameter]=NoScale(), t:list[TranslationParameter]=NoTranslation()):
        super().__init__(r,s,t)

class AffineTransformationSet(BaseAffineTransformationSet):
    def __init__(self,  r:list[RotationParameter]=NoRotation(), s:list[ScaleParameter]=NoScale(), t:list[TranslationParameter]=NoTranslation()):
        super().__init__(r,s,t)

    def make_transformation(self, ap:affine.AffineParameters):
        return AffineTransformation(ap)

    def copy(self):
        return AffineTransformationSet(self.r,self.s,self.t)


class RotationTransformationSet(BaseAffineTransformationSet):
    def __init__(self, r:list[RotationParameter]):
        super().__init__(r=r)

    def make_transformation(self, ap:affine.AffineParameters):
        return RotationTransformation(ap.r)

    def copy(self):
        return RotationTransformationSet(self.r)

class ScaleTransformationSet(BaseAffineTransformationSet):
    def __init__(self, s:list[ScaleParameter]):
        super().__init__(s=s)

    def make_transformation(self, ap:affine.AffineParameters):
        return ScaleTransformation(ap.s)

    def copy(self):
        return ScaleTransformationSet(self.s)

class TranslationTransformationSet(BaseAffineTransformationSet):
    def __init__(self, t:list[TranslationParameter]):
        super().__init__(t=t)

    def make_transformation(self, ap:affine.AffineParameters):
        return TranslationTransformation(ap.t)

    def copy(self):
        return TranslationTransformationSet(self.t)
