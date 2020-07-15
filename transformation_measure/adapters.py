import abc
import torch

class TransformationAdapter:

    @abc.abstractmethod
    def pre_adapt(self,x):
        pass
    def post_adapt(self,x):
        pass


class PytorchNumpyImageTransformationAdapter(TransformationAdapter):
    def __init__(self,use_cuda):
        self.use_cuda=use_cuda
    # adapt a pytorch batch of images to process in numpy
    def pre_adapt(self,x):
        x = x.cpu().numpy()
        x= x.transpose((0,2, 3, 1))
        return x

    # adapt a numpy batch of images to process pytorch
    def post_adapt(self,x):
        x = x.transpose(0,3, 1, 2)
        x = torch.from_numpy(x)
        if self.use_cuda:
            x=x.cuda()
        return x




class NumpyPytorchImageTransformationAdapter(TransformationAdapter):
    def __init__(self,use_cuda):
        self.use_cuda=use_cuda
        self.inverse=PytorchNumpyImageTransformationAdapter(use_cuda)

    def pre_adapt(self,x):
        return self.inverse.post_adapt(x)

    def post_adapt(self,x):
        return self.inverse.pre_adapt(x)