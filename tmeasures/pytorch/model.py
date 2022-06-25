

from torch import full, nn
import torch
import abc
import typing
from typing import List,Tuple,Dict,Callable, Union,MutableMapping,Any

class ActivationsModule(nn.Module):

    @abc.abstractmethod
    def activation_names(self) -> List[str]:
        raise NotImplementedError()

    @abc.abstractmethod
    def forward_activations(self, args) -> List[torch.Tensor]:
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
    def forward_activations(self, args) ->  List[torch.Tensor]:
        y,activations = self.inner_model.forward_activations(args)
        activations = [activations[i] for i in self.indices]
        return y,activations



def intersect_lists(a:List,b:List):
    return list(set(a) & set(b))

class DuplicateKeyError(ValueError):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

Input = Union[MutableMapping,Any]
def flatten_dict(d_or_val:Input, prefix='', sep='/', allow_repeated=False)->Dict[str,Any]:
    print(d_or_val)
    if isinstance(d_or_val, MutableMapping):
        result = {}
        if prefix =="":
            prefix_sep = ""
        else:
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

def flatten_dict_list(d_or_val:Input,key="",full_name=True)->List[Tuple[str,Any]]:
    if isinstance(d_or_val, MutableMapping):
        result = []
        for k, v in d_or_val.items():
            if full_name:
                new_key = key + "/" + k
            else:
                new_key = k
            flattened_child = flatten_dict_list(v,key=new_key)
            result+=flattened_child
        return result
    else :
        return [(key, d_or_val)]


def named_children_deep(m: torch.nn.Module):
    children = dict(m.named_children())
    if children == {}:
        return m
    else:
        output = {}
        for name, child in children.items():
            if name.isnumeric():
                key = child.__class__.__name__+"_"+name
            else:
                key=name
            if key in output:
                raise DuplicateKeyError
            try:
                output[key] = named_children_deep(child)
            except TypeError:
                output[key] = named_children_deep(child)
        return output

class AutoActivationsModule(ActivationsModule):
    def __init__(self,module:nn.Module,full_name=True) -> None:
        super().__init__()
        self.reset_values()
        self.module = module
        self.children_tree = named_children_deep(module)
        
        self.names,self.activations = zip(*flatten_dict_list(self.children_tree,full_name=full_name))
        

        self.register_hooks(self.activations)
        

    def register_hooks(self,activations:List[nn.Module]):
        def store_activation(index):
            def hook(model, input, output):
                self.values[index] = output.detach()
            return hook

        for i,v in enumerate(activations):
            v.register_forward_hook(store_activation(i))

    def reset_values(self):
        self.values = {} 

    def forward_activations(self, args) -> List[torch.Tensor]:
        '''
        This function is not thread safe.
        '''
        # clear values
        self.reset_values()
        # call model, hooks are executed
        self.module.forward(args)
        # transform to list and ensure order of values is same as order of activation names
        activations_list = [self.values[i] for i in range(len(self.activations))]
        return activations_list

    def activation_names(self) -> List[str]:
        return self.names