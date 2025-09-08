from .. import logger as tm_logger
from .base import PyTorchLayerMeasure, STMatrixIterator
from .stats_running import RunningMeanAndVarianceWelford, RunningMeanWelford

logger = tm_logger.getChild("pytorch.layer_measures")


class Variance(PyTorchLayerMeasure):
    def eval(self, st_iterator: STMatrixIterator, layer_name: str):
        mean = RunningMeanWelford()

        for row, row_iterator in enumerate(st_iterator):
            logger.info(f"frow {layer_name}, row {row}/{len(st_iterator)}")

            row_variance = RunningMeanAndVarianceWelford()
            for batch_i, batch_activations in enumerate(row_iterator):
                logger.info(f"fcol {layer_name}, row {row}/{len(st_iterator)}, batch_i {batch_i}/{len(row_iterator)}")
                row_variance.update_batch(batch_activations.double())
            row_std = row_variance.std()
            mean.update(row_std)
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
