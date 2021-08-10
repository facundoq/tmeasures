from transformational_measures.pytorch.base import PyTorchLayerMeasure, STMatrixIterator
from transformational_measures.pytorch.stats_running import RunningMeanWelford, RunningMeanAndVarianceWelford


class Variance(PyTorchLayerMeasure):
    def eval(self, st_iterator:STMatrixIterator):
        mean = RunningMeanWelford()
        for row,row_iterator in enumerate(st_iterator):
            row_variance = RunningMeanAndVarianceWelford()
            for batch_n,batch_activations in enumerate(row_iterator):
                row_variance.update_batch(batch_activations.double())
            #print(row_variance.m.device(),row_variance.s.device(),mean.mu.device())
            row_std = row_variance.std()
            mean.update(row_std)
            # print(row_variance.m.device, row_variance.s.device, mean.mu.device)
        return mean.mean()


# class Distance(PyTorchLayerMeasure):
#     def eval(self, st_iterator:STMatrixIterator):
#         mean = RunningMeanWelford()
#         for row,row_iterator in enumerate(st_iterator):
#             row_variance = RunningMeanAndVarianceWelford()
#             for batch_n,batch_activations in enumerate(row_iterator):
#                 row_variance.update_all(batch_activations.double())
#             row_std = row_variance.std()
#             mean.update(row_std)
#         return mean.mean()
#
