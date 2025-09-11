import torch
import tmeasures as tm

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