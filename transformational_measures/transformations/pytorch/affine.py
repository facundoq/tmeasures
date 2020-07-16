import torch
from torch.nn import functional as F
from typing import List, Iterator
from ..affine import RotationParameter,ScaleParameter,TranslationParameter
from .. import affine
import numpy as np


class AffineTransformation(affine.AffineTransformation):

    def __init__(self,ap:affine.AffineParameters):
        super().__init__(ap)
        self.transformation_matrix= self.get_transformation_matrix()
        self.transformation_matrix= self.transformation_matrix.unsqueeze(0)



    def get_transformation_matrix(self,)->torch.Tensor:
        # center_matrix=torch.eye(3)
        # # center_matrix[:2,2]=-.5
        # decenter_matrix=torch.eye(3)
        # # decenter_matrix[:2,2]=.5

        r,s,t=self.ap
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
            n, c, h, w = x.shape
            # print(x.shape)
            # print(self.transformation_matrix.shape)
            shape=[1,c,h,w]
            grid = F.affine_grid(self.transformation_matrix,shape ,align_corners=False)

            #if self.use_cuda:
            grid=grid.to(x.device)
            grid = grid.expand(n,*grid.shape[1:])
            x = F.grid_sample(x, grid,align_corners=False,padding_mode="border")
        return x

    def inverse(self):
        return AffineTransformation(self.ap.inverse())




class AffineGenerator(affine.AffineGenerator):
    def __init__(self, r:Iterator[RotationParameter]=None, s:Iterator[ScaleParameter]=None, t:Iterator[TranslationParameter]=None):
        super().__init__(r,s,t)


    def make_transformation(self, ap:affine.AffineParameters):
        return AffineTransformation(ap)

    def copy(self):
        return AffineGenerator(self.r,self.s,self.t)

