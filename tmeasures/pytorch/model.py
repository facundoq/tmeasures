

from torch import nn
import torch
import abc
import typing
from typing import List,Tuple,Dict,Callable, Union,MutableMapping,Any

class ActivationsModule(nn.Module):

    @abc.abstractmethod
    def activation_names(self) -> List[str]:
        raise NotImplementedError()

    @abc.abstractmethod
    def forward_activations(self, args) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        raise NotImplementedError()

    def n_activations(self):
        return len(self.activation_names())
#


ActivationFilter = Callable[[ActivationsModule,str],bool]
class FilteredActivationsModule(ActivationsModule):
    def __init__(self, inner_model:ActivationsModule, activations_filter:ActivationFilter):
        super().__init__()
        self.inner_model = inner_model
        original_names = inner_model.activation_names()
        self.names = [name for name in original_names if activations_filter(inner_model,name)]
        assert len(self.names)>0
        self.indices = [original_names.index(n) for n in self.names]
    def forward(self):
        return self.inner_model.forward()
    def activation_names(self) -> List[str]:
        return self.names
    def eval(self):
        self.inner_model.eval()
    @abc.abstractmethod
    def forward_activations(self, args) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        y,activations = self.inner_model.forward_activations(args)
        activations = [activations[i] for i in self.indices]
        return y,activations



def intersect_lists(a:List,b:List):
    return list(set(a) & set(b))

class DuplicateKeyError(ValueError):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

Input = Union[MutableMapping,Any]
def flatten_dict(d_or_val:Input, prefix='', sep='_', allow_repeated=False)->Dict[str,Any]:
    if isinstance(d_or_val, MutableMapping):
        result = {}
        prefix_sep = prefix+sep
        for k, v in d_or_val.items():
            new_prefix = prefix_sep+str(k)
            flattened_child = flatten_dict(v,new_prefix,sep)
            if not allow_repeated :
                intersection = intersect_lists(flattened_child.keys(),d_or_val.keys())
                if len(intersection)>0:
                    raise DuplicateKeyError(intersection)
            result.update(flattened_child)
        return result
    else :
        return { prefix : d_or_val }



def named_children_deep(m: torch.nn.Module,prefix=""):
    children = dict(m.named_children())
    if children == {}:
        return m
    else:
        output = {}
        for name, child in children.items():
            if name in output:
                raise DuplicateKeyError
            try:
                output[name] = named_children_deep(child)
            except TypeError:
                output[name] = named_children_deep(child)
        return output

class AutoActivationsModule(ActivationsModule):
    def __init__(self,module:nn.Module) -> None:
        super().__init__()
        self.module = module
        self.children_tree = named_children_deep(module)
        self.children = flatten_dict(self.children_tree)
        self.register_hooks(self.children)

    def register_hooks(self,children:Dict[str,nn.Module]):
        def store_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook

        for k,v in children.items():
            v.register_forward_hook(store_activation(k))

    def forward_activations(self, args) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        '''
        This function is not thread safe.
        '''
        # reset activations
        self.activations = {}    
        # call model, hooks are executed
        y = self.module.forward(args)
        # transform to list and ensure order of values is same as order of activation names
        activations_list = [self.activations[k] for k in self.activation_names()]
        return y, activations_list

    def activation_names(self) -> List[str]:
        return list(self.children.keys())