from __future__ import annotations

import abc
from ast import Module
from typing import Any, Callable, List, MutableMapping, Tuple, TypeAlias, Union

import torch
from torch import nn

from ..utils import DuplicateKeyError, Graph, flatten_dict_list, get_all

ActivationValues:TypeAlias = dict[str,torch.Tensor]
FlatActivations:TypeAlias = dict[str,nn.Module]

class ActivationsModule(abc.ABC):

    @abc.abstractmethod
    def activation_names(self) -> List[str]:
        raise NotImplementedError()


    @property
    @abc.abstractmethod
    def activations(self) -> FlatActivations:
        raise NotImplementedError()

    @abc.abstractmethod
    def forward_activations(self, args) -> List[torch.Tensor]:
        raise NotImplementedError()

    def n_activations(self):
        return len(self.activation_names())

class ManualActivationsModule(ActivationsModule):
    def __init__(self,module:nn.Module,activations:FlatActivations):
        super().__init__()
        self._activations = activations
        self.module=module
        self.register_hooks(activations)

    @property
    def activations(self):
        return self._activations

    def register_hooks(self,activations:FlatActivations):
        self.reset_values()
        def store_activation(key):
            def hook(model, input, output):
                self.values[key] = output.detach()
            return hook

        for k, v in activations.items():
            v.register_forward_hook(store_activation(k))

    def reset_values(self):
        self.values:ActivationValues = {}

    def forward_activations(self, args) -> ActivationValues:
        '''
        This function is not thread safe.
        '''
        # clear values
        self.reset_values()
        # call model, hooks are executed
        self.module.forward(args)
        self.check_errors()
        # transform to list and ensure order of values is same as order of activation names
        return self.values

    def check_errors(self):
        # Check that all activations are present in the values
        nv,na = len(self.values),len(self.activations)
        if nv!=na:
            indices_a = set(list(self.activations.keys()))
            indices_v = set(list(self.values.keys()))
            diff = indices_a-indices_v
            def diff_str(): return '\n'.join(diff)
            assert len(diff)==0, f"Values output by network ({nv}) dont match original number of activations ({na}). Ensure all modules produce outputs with the same shape on each iteration. Activations missing: \n {diff_str()}. "

    def activation_names(self) -> List[str]:
        return list(self.activations.keys())

ActivationFilter = Callable[[str,nn.Module],bool]

def get_activations(module:nn.Module,full_name=True,separator="_",)->FlatActivations:
    children_tree = named_children_deep(module,separator)
    activations = dict(flatten_dict_list(children_tree,full_name=full_name))
    return activations

def named_children_deep(m: torch.nn.Module,separator:str="_")->Graph[nn.Module]:
        children = dict(m.named_children())
        if children == {}:
            return m
        else:
            output = {}
            for name, child in children.items():
                if name.isnumeric():
                    key = child.__class__.__name__+separator+name
                else:
                    key=name
                if key in output:
                    raise DuplicateKeyError
                try:
                    output[key] = named_children_deep(child,separator)
                except TypeError:
                    output[key] = named_children_deep(child,separator)
            return output

class AutoActivationsModule(ManualActivationsModule):
    def __init__(self,module:nn.Module,full_name=True,separator="_",filter:ActivationFilter=lambda x,y: True) -> None:
        activations = get_activations(module,full_name=full_name,separator=separator)
        activations = {k:v for k,v in activations.items() if filter(k,v)}
        super().__init__(module,activations)


