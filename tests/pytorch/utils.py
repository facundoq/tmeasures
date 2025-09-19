from attr import dataclass
import numpy as np
import torch
import tmeasures as tm
from numpy.testing import assert_allclose

@dataclass
class MeasureFixture:
    model:torch.nn.Module
    measure:tm.pytorch.PyTorchMeasure
    expected_result:list[np.ndarray]
    dataset:torch.utils.data.Dataset
    transformations:tm.pytorch.transformations.PyTorchTransformationSet
    options:tm.pytorch.PyTorchMeasureOptions = tm.pytorch.PyTorchMeasureOptions(batch_size=32,num_workers=0)

    def assert_fixture(self,atol=1e-5):
        activations_model = tm.pytorch.AutoActivationsModule(self.model)
        result = self.measure.eval(self.dataset,self.transformations,activations_model,self.options)
        
        result = result.numpy()
        for name,layer,expected_layer in zip(result.layer_names,result.layers,self.expected_result):
            assert_allclose(layer,expected_layer,err_msg=f"Error in {self.measure} for activation '{name}'",atol=atol)
        
class ConstantModel(torch.nn.Module):
    def __init__(self,value=torch.Tensor(0)) -> None:
        super().__init__()
        self.value=value
    def forward(self,x:torch.Tensor):
        n = x.shape[0]
        result =  self.value.expand(n,*self.value.shape)
        return result
    
class IdentityModel(torch.nn.Module):
    def __init__(self,) -> None:
        super().__init__()
    def forward(self,x:torch.Tensor):
        return x
    
class RandomModel(torch.nn.Module):
    def __init__(self,shape:tuple,mean=0.0,std=1.0) -> None:
        super().__init__()
        self.mean=mean
        self.std=std
        self.shape=shape
    def forward(self,x:torch.Tensor):
        n = x.shape[0]
        shape = (n,*self.shape)
        return torch.normal(mean=self.mean,std=self.std,size=shape)
        


class ConstantDataset(torch.utils.data.Dataset):
    def __init__(self, n:int,value=torch.tensor((0,)) ):
        super().__init__()
        self.value = value
        self.n = n
    def __len__(self):
        return self.n
    def __getitem__(self, index):
        return self.value
    

class RepeatedIdentitySet(tm.pytorch.transformations.PyTorchTransformationSet):
    def __init__(self,transformations=1):
        super().__init__([tm.pytorch.transformations.IdentityTransformation()]*transformations)
    def valid_input(self):
        return True
    def copy(self):
        return self
    def id(self):
        return "Identity"