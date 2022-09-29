from __future__ import annotations

from typing import Tuple
import torch
from tmeasures import TransformationSet
import tmeasures as tm

def list_get_all(list:list,indices:list[int])->[]:
    return [list[i] for i in indices]


class ActivationsTransformer:
    def __init__(self,activation_shapes: [Tuple[int,]], layer_names: [str],
                         transformation_set: tm.TransformationSet,inverse:bool):

        self.layer_names,self.indices=self.get_valid_layers(activation_shapes,layer_names,transformation_set)


        self.valid_shapes=list_get_all(activation_shapes,self.indices)

        self.transformation_sets=self.get_transformations_set(self.valid_shapes, transformation_set, inverse)
        assert len(self.transformation_sets) == len(self.valid_shapes)
        m= len(transformation_set)
        for s in self.transformation_sets:
            mi=len(s)
            assert mi == m, f"Transformation sets for each layer should have the same number of transformations ({mi}) as the original transformation set ({m})"



    def get_valid_layers(self, activation_shapes: [Tuple[int,]], layer_names: [str],
                         transformation_set: tm.TransformationSet)->([str],[int]):
        # get indices of layers for which the transformation is valid
        indices = [i for i, shape in enumerate(activation_shapes) if transformation_set.valid_input(shape)]
        # keep only this layers
        layer_names = list_get_all(layer_names, indices)

        return layer_names, indices

    def get_transformations_set(self, shapes: [Tuple[int,]],
                                transformation_set: tm.TransformationSet,inverse:bool):
        transformation_sets = []

        for s in shapes:
            n, c, h, w = s
            layer_transformation_set: tm.TransformationSet = transformation_set.copy()
            # print(len(transformation_set),"vs",len(layer_transformation_set))
            # layer_transformation_set.set_pytorch(False)
            #layer_transformation_set.set_input_shape((h, w, c))
            layer_transformation_set_list = list(layer_transformation_set)
            if inverse:
                layer_transformation_set_list = [l.inverse() for l in layer_transformation_set_list]
            transformation_sets.append(layer_transformation_set_list)
        return transformation_sets

    def filter_activations(self,activations: [torch.Tensor])->[torch.Tensor]:
        return list_get_all(activations,self.indices)

    def trasform_st_same_row(self, activations: [torch.Tensor],
                             t_start: int, t_end: int):
        # iterate over each layer and corresponding layer transformations
        for layer_activations, layer_transformations in zip(activations, self.transformation_sets):
            # each sample of the layer activations corresponds to a different column of the st matrix
            # => a different transformation
            # t_start and t_end indicate the corresponding column indices
            for i, transformation in enumerate(layer_transformations[t_start:t_end]):
                transformed_activations = transformation(layer_activations[i,:])
                # print(fm.shape, inverse_fm.shape)
                layer_activations[i,:] = transformed_activations

    def trasform_st_same_column(self, activations: [torch.Tensor], t_i: int):
        for layer_activations, layer_transformations in zip(activations, self.transformation_sets):
            # each sample of the layer activations corresponds to a different row of the st matrix
            # => a different sample
            # t_i indicate the corresponding column index, that is, the transformation index
            transformation = layer_transformations[t_i]
            for i in range(layer_activations.shape[0]):
                layer_activations[i,:] = transformation(layer_activations[i,:])

