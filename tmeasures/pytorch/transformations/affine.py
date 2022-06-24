import torch
from torch.nn import functional as F
from typing import List, Iterator
from tmeasures.transformations.affine import RotationParameter,ScaleParameter,TranslationParameter
from tmeasures.transformations import affine
import numpy as np
import math

from tmeasures.pytorch.transformations import PyTorchTransformation, PyTorchTransformationSet


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

        r,s,t=self.ap
        r= r*180
        sx,sy=s
        #tx,ty=t
        matrix = torch.eye(3)
        matrix[:2, :2] = torch.tensor([[np.cos(r)/sx, -1.0*np.sin(r)/sy],
                                        [np.sin(r)/sx, np.cos(r)/sy]])
        matrix[:2,2]=torch.tensor(t)*2
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
        p=affine.AffineParameters(r=p)
        super().__init__(p)

    def parameters(self):
        # encode with cos and sin
        # https://arxiv.org/pdf/1702.01499.pdf0
        return torch.tensor([math.cos(self.p*360),math.sin(self.p*360)])

    def inverse(self):
        return RotationTransformation(-self.p)

class ScaleTransformation(AffineTransformation):
    def __init__(self,p:affine.ScaleParameter):
        self.p=p
        p=affine.AffineParameters(s=p)
        super().__init__(p)

    def parameters(self):
        sx,sy=self.p
        return torch.tensor([sx,sy])
    def inverse(self):
        sx,sy=self.p
        return ScaleTransformation((1/sx,1/sy))

class TranslationTransformation(AffineTransformation):
    def __init__(self,p:affine.TranslationParameter):
        self.p=p
        p=affine.AffineParameters(t=p)
        super().__init__(p)
    def parameters(self):
        tx, ty = self.p
        return torch.tensor([tx, ty])

    def inverse(self):
        tx,ty=self.p
        return TranslationTransformation((-tx,-ty))

class BaseAffineTransformationGenerator(affine.AffineGenerator,PyTorchTransformationSet):
    def __init__(self, r:Iterator[RotationParameter]=None, s:Iterator[ScaleParameter]=None, t:Iterator[TranslationParameter]=None):
        super().__init__(r,s,t)

class AffineGenerator(BaseAffineTransformationGenerator):
    def __init__(self, r:Iterator[RotationParameter]=None, s:Iterator[ScaleParameter]=None, t:Iterator[TranslationParameter]=None):
        super().__init__(r,s,t)

    def make_transformation(self, ap:affine.AffineParameters):
        return AffineTransformation(ap)

    def copy(self):
        return AffineGenerator(self.r,self.s,self.t)


class RotationGenerator(BaseAffineTransformationGenerator):
    def __init__(self, r:Iterator[RotationParameter]):
        super().__init__(r,None,None)

    def make_transformation(self, ap:affine.AffineParameters):
        return RotationTransformation(ap.r)

    def copy(self):
        return RotationGenerator(self.r)

class ScaleGenerator(BaseAffineTransformationGenerator):
    def __init__(self, s:Iterator[ScaleParameter]):
        super().__init__(None,s,None)

    def make_transformation(self, ap:affine.AffineParameters):
        return ScaleTransformation(ap.s)

    def copy(self):
        return ScaleGenerator(self.s)

class TranslationGenerator(BaseAffineTransformationGenerator):
    def __init__(self, t:Iterator[TranslationParameter]):
        super().__init__(None,None,t)

    def make_transformation(self, ap:affine.AffineParameters):
        return TranslationTransformation(ap.t)

    def copy(self):
        return TranslationGenerator(self.t)