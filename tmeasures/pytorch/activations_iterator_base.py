from tmeasures.np.activations_iterator import ActivationsIterator
from .model import ActivationsModule
import torch
from torch.utils.data import DataLoader,Dataset

from tmeasures import TransformationSet, Transformation
from tmeasures.adapters import TransformationAdapter


import tmeasures as tm
import abc
from .activations_transformer import ActivationsTransformer




class PytorchActivationsIterator(ActivationsIterator):

    def __init__(self, model: ActivationsModule, dataset:Dataset, transformations: TransformationSet, batch_size,
                 num_workers, use_cuda,adapter: TransformationAdapter=None):

        self.model = model
        self.dataset = dataset
        self.transformations = transformations

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_cuda = use_cuda
        self.adapter = adapter

    @abc.abstractmethod
    def samples_activation(self, t_i: int, transformation: Transformation, dataloader: DataLoader):
        pass

    @abc.abstractmethod
    def transformations_activations(self, x: torch.Tensor):
        pass

    def get_transformations(self) ->tm.TransformationSet:
        return self.transformations
    def transformations_first(self):
        for t_i, transformation in enumerate(self.transformations):
            dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False,num_workers=self.num_workers, drop_last=False, pin_memory=True)
            yield transformation, self.samples_activation(t_i, transformation, dataloader)

    '''
         Returns the activations of the models by iterating first over transformations and 
         then, for each transformation, over samples
     '''

    def samples_first(self):
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False,
                                num_workers=self.num_workers, drop_last=False)

        with torch.no_grad():
            for batch in dataloader:
                batch_cpu = batch
                if self.use_cuda:
                    batch = batch.cuda()
                for i in range(batch.shape[0]):
                    x = batch[i, :]
                    yield batch_cpu[i, :], self.transformations_activations(x)

    def transform_sample(self, x: torch.Tensor):
        x = x.unsqueeze(0)
        results = []
        for i, transformation in enumerate(self.transformations):
            transformed = self.transform_batch(transformation, x)
            results.append(transformed)
        result = torch.cat(results)
        return result

    def transform_batch(self, transformation, x: torch.Tensor):
        if not self.adapter is None:
            x = self.adapter.pre_adapt(x)
        results = []
        for i in range(x.shape[0]):
            results.append(transformation(x[i,:]))
        x = torch.stack(tuple(results),dim=0)

        if not self.adapter is None:
            x = self.adapter.post_adapt(x)
        return x

    def layer_names(self):
        return self.model.activation_names()

    def get_inverted_activations_iterator(self) -> ActivationsIterator:
        return InvertedPytorchActivationsIterator(self.model, self.dataset, self.transformations, self.batch_size,
                                                  self.num_workers,  self.use_cuda,self.adapter)

    def get_both_iterator(self) -> ActivationsIterator:
        return BothPytorchActivationsIterator(self.model, self.dataset, self.transformations, self.batch_size,
                                              self.num_workers,  self.use_cuda,self.adapter,)

    def get_normal_activations_iterator(self) -> ActivationsIterator:
        return NormalPytorchActivationsIterator(self.model, self.dataset, self.transformations, self.batch_size,
                                                self.num_workers, self.use_cuda,self.adapter,)


class NormalPytorchActivationsIterator(PytorchActivationsIterator):

    def samples_activation(self, t_i, transformation, dataloader):
        for batch in dataloader:
            if self.use_cuda:
                batch = batch.cuda()
            batch = self.transform_batch(transformation, batch)
            with torch.no_grad():
                batch_activations = self.model.forward_activations(batch)
                batch_activations = [a.cpu().numpy() for a in batch_activations]
            yield batch, batch_activations

    def transformations_activations(self, x):

        x_transformed = self.transform_sample(x)
        dataloader = DataLoader(x_transformed, batch_size=self.batch_size, shuffle=False,
                                num_workers=0, drop_last=False)
        for batch in dataloader:
            with torch.no_grad():
                batch_activations = self.model.forward_activations(batch)
            batch_activations = [a.cpu().numpy() for a in batch_activations]
            yield batch, batch_activations


class InvertedPytorchActivationsIterator(PytorchActivationsIterator):

    def samples_activation(self, t_i, transformation, dataloader):
        self.activations_transformer = None
        for batch in dataloader:
            if self.use_cuda:
                batch = batch.cuda()
            batch = self.transform_batch(transformation, batch)
            with torch.no_grad():
                batch_activations = self.model.forward_activations(batch)
                if self.activations_transformer is None:
                    shapes = [a.shape for a in batch_activations]
                    self.activations_transformer = ActivationsTransformer(shapes, self.model.activation_names(),
                                                                          self.get_transformations(), inverse=True)
                # filter activations to those accepted by the transformations
                batch_activations = self.activations_transformer.filter_activations(batch_activations)

                # inverse transform selected activations
                self.activations_transformer.trasform_st_same_column(batch_activations, t_i)
                batch_activations = [a.cpu().numpy() for a in batch_activations]
                yield batch, batch_activations

    '''
         Returns the activations of the models by iterating first over transformations and 
         then, for each transformation, over samples
     '''

    def transformations_activations(self, x):
        x_transformed = self.transform_sample(x)
        dataloader = DataLoader(x_transformed, batch_size=self.batch_size, shuffle=False,
                                num_workers=0, drop_last=False)
        self.activations_transformer = None
        t_start = 0
        for batch in dataloader:
            with torch.no_grad():
                batch_activations = self.model.forward_activations(batch)
            t_end = t_start + batch.shape[0]
            if self.activations_transformer is None:
                shapes = [a.shape for a in batch_activations]
                self.activations_transformer = ActivationsTransformer(shapes, self.model.activation_names(), self.transformations, inverse=True)
            # filter activations to those accepted by the transformations
            batch_activations = self.activations_transformer.filter_activations(batch_activations)
            # inverse transform selected activations
            self.activations_transformer.trasform_st_same_row(batch_activations, t_start, t_end)
            batch_activations = [a.cpu().numpy() for a in batch_activations]
            t_start = t_end
            yield batch, batch_activations

    def get_inverted_activations_iterator(self) -> ActivationsIterator:
        return self

    def layer_names(self) -> [str]:
        return self.activations_transformer.layer_names


class BothPytorchActivationsIterator(PytorchActivationsIterator):

    def samples_activation(self, t_i, transformation, dataloader):
        for batch in dataloader:
            if self.use_cuda:
                batch = batch.cuda()

            transformed_batch = self.transform_batch(transformation, batch)
            with torch.no_grad():
                pre_transformed_activations = self.model.forward_activations(transformed_batch)
                post_transformed_activations = self.model.forward_activations(batch)
            if self.activations_transformer is None:
                shapes = [a.shape for a in pre_transformed_activations]
                self.activations_transformer = ActivationsTransformer(shapes, self.model.activation_names(),
                                                                      self.get_transformations(),False)
            # filter activations to those accepted by the transformations
            pre_transformed_activations = self.activations_transformer.filter_activations(pre_transformed_activations)
            post_transformed_activations = self.activations_transformer.filter_activations(post_transformed_activations)
            # print([a.shape for a in post_transformed_activations])
            # transform selected activations
            self.activations_transformer.trasform_st_same_column(post_transformed_activations, t_i)
            # print([a.shape for a in post_transformed_activations])
            pre_transformed_activations = [a.cpu().numpy() for a in pre_transformed_activations]
            post_transformed_activations = [a.cpu().numpy() for a in post_transformed_activations]
            yield batch, pre_transformed_activations, post_transformed_activations

    '''
         Returns the activations of the models by iterating first over transformations and 
         then, for each transformation, over samples
     '''

    def transformations_activations(self, x):
        x_transformed = self.transform_sample(x)
        dataloader = DataLoader(x_transformed, batch_size=self.batch_size, shuffle=False,
                                num_workers=0, drop_last=False)
        t_start = 0
        # calculate activations of untransformed sample
        with torch.no_grad():
            untransformed_activations = self.model.forward_activations(x.unsqueeze(0))
        # Generate activations transformers
        shapes = [a.shape for a in untransformed_activations]
        self.activations_transformer = ActivationsTransformer(shapes, self.model.activation_names(), self.transformations,False)

        # remove activations that can't be transformed
        untransformed_activations = self.activations_transformer.filter_activations(untransformed_activations)

        for batch in dataloader:
            with torch.no_grad():
                pre_transformed_activations = self.model.forward_activations(batch)

            b_n=batch.shape[0]
            t_end = t_start + b_n
            # filter activations to those accepted by the transformations
            pre_transformed_activations = self.activations_transformer.filter_activations(pre_transformed_activations)
            # calculate post transformed activations
            # post_transformed_activations = [a.clone() for a in untransformed_activations]
            # replicate to batch size b_n
            post_transformed_activations = [a.expand(b_n,*a.shape[1:]).clone() for a in untransformed_activations]
            # inverse transform selected activations
            self.activations_transformer.trasform_st_same_row(post_transformed_activations, t_start, t_end)
            t_start = t_end
            pre_transformed_activations = [a.cpu().numpy() for a in pre_transformed_activations]
            post_transformed_activations = [a.cpu().numpy() for a in post_transformed_activations]
            yield batch, pre_transformed_activations, post_transformed_activations

    def get_both_iterator(self) -> ActivationsIterator:
        return self

    def layer_names(self) -> [str]:
        return self.activations_transformer.layer_names

