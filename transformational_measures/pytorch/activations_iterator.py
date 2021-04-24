from .dataset2d import STDataset
import torch

class PytorchActivationsIterator:

    def __init__(self, model: ObservableLayersModule, dataset:STDataset, batch_size,
                 num_workers, device:torch.device=torch.device("cpu"), adapter: TransformationAdapter=None):

        self.model = model
        self.dataset = dataset
        self.device=device
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.adapter = adapter



class Measure:
    pass