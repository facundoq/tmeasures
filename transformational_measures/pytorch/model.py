from torch import nn
import torch
import abc
import typing

class ObservableLayersModule(nn.Module):

    @abc.abstractmethod
    def activation_names(self) -> typing.List[str]:
        raise NotImplementedError()

    @abc.abstractmethod
    def forward_intermediates(self, args) -> (torch.Tensor, typing.List[torch.Tensor]):
        raise NotImplementedError()

    def n_intermediates(self):
        return len(self.activation_names())
#
# class ActivationsSelector(abc.ABC):
#     @abc.abstractmethod
#     def select_layers(self,all_layers:[str]):
#         pass
#
# class AllLayers(ActivationsSelector):
#     def select_layers(self,all_layers:[str]):
#         return all_layers
#
# class FilterActivations(ActivationsSelector):
#     def __init__(self,filter:typing.Callable[[str],bool]):
#         self.filter = filter
#     def select_layers(self,all_layers:[str]):
#         return list(filter(self.filter,all_layers))
#
# class SubsetActivations(FilterActivations):
#     def __init__(self,layers:[str]):
#         self.activations = layers
#         filter = lambda x: x in layers
#         super().__init__(filter)

ActivationFilter = typing.Callable[[ObservableLayersModule,str],bool]
class FilteredActivationsModel(ObservableLayersModule):
    def __init__(self, inner_model:ObservableLayersModule, activations_filter:ActivationFilter):
        super().__init__()
        self.inner_model = inner_model
        original_names = inner_model.activation_names()
        self.names = [name for name in original_names if activations_filter(inner_model,name)]
        assert len(self.names)>0
        self.indices = [original_names.index(n) for n in self.names]
    def forward(self):
        return self.inner_model.forward()
    def activation_names(self) -> typing.List[str]:
        return self.names
    def eval(self):
        self.inner_model.eval()
    @abc.abstractmethod
    def forward_intermediates(self, args) -> (torch.Tensor, typing.List[torch.Tensor]):
        y,activations = self.inner_model.forward_intermediates(args)
        activations = [activations[i] for i in self.indices]
        return y,activations


