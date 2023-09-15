from typing import Any, List
from . import ActivationsModule
from torch import nn
import torch
class SequentialWithIntermediates(nn.Sequential, ActivationsModule):
    def __init__(self,*args):
        super(SequentialWithIntermediates, self).__init__(*args)

    def forward_intermediates(self,input_tensor)->(List[torch.Tensor]):
        submodules=self._modules.values()
        if len(submodules)==0:
            return  input_tensor,[input_tensor]

        outputs=[]
        for module in submodules:
            if isinstance(module, ActivationsModule):
                intermediates=module.forward_activations(input_tensor)
                outputs+=(intermediates)
            else:
                input_tensor= module(input_tensor)
                outputs.append(input_tensor)
        return outputs

    def activation_names(self)->List[str]:
        submodules = self._modules.values()
        if len(submodules) == 0:
            return ["identity"]
        if len(submodules) == 1:
            module = list(submodules)[0]

            if isinstance(module, ActivationsModule):
                return ["0_"+name for name in module.activation_names()]
            else:
                name = module.__class__.__name__
                return [self.abbreviation(name)]

        # len(submodules)>1
        outputs = []
        index=0

        for module in submodules:
            if isinstance(module, ActivationsModule):
                index += 1
                module_name=self.abbreviation(module.__class__.__name__)
                names=[f"{module_name}{index}_{name}" for name in module.activation_names()]
                outputs +=names
            else:
                name=module.__class__.__name__
                if name.startswith("Conv") or name.startswith("Linear"):
                    index += 1  # conv and fc layers increase index
                name=f"{index}{self.abbreviation(name)}"
                outputs.append(name)
        return outputs

    def abbreviation(self, name:str)->str:
        if name.startswith("Conv"):
            name = "c"
        elif name.startswith("BatchNorm"):
            name = "bn"
        elif name.startswith("ELU"):
            name = "elu"
        elif name.startswith("ReLU"):
            name= "relu"
        elif name.startswith("Linear"):
            name = "fc"
        elif name.startswith("Add"):
            name = "+"
        elif "Softmax" in name:
            name="sm"
        elif name == "Sequential":
            name = ""
        elif name == "SequentialWithIntermediates":
            name = ""
        elif name == "Block":
            name = "b"
        return name
    
import ctypes, os, threading
def set_thread_name_np(the_name):
    the_lib_path = "/lib/libpthread-2.42.so"
    if not os.path.isfile(the_lib_path):
        return None
    try:
        libpthread = ctypes.CDLL(the_lib_path)
    except:
        return None
    if hasattr(libpthread, "pthread_setname_np"):
        pthread_setname_np = libpthread.pthread_setname_np
        pthread_setname_np.argtypes = [ctypes.c_void_p,
                                       ctypes.c_char_p]
        pthread_setname_np.restype = ctypes.c_int
        if isinstance(the_name, str):
            the_name = the_name.encode('ascii', 'replace')
        if type(the_name) is not bytes:
            return None
        the_thread = threading.current_thread()
        ident = getattr(the_thread, "ident", None)
        if ident is not None:
            pthread_setname_np(ident, the_name[:15])
            return True
    return None    