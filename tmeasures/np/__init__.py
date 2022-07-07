
"""
The :mod:`.pytorch` module contains implementations of the measures for the PyTorch framework, as well as some utility functions.

A :class:`.PyTorchMeasure` can be evaluated with a PyTorch :class:`.ActivationsModule`, dataset and :class:`.PyTorchTransformationSet`. The evaluation of the measure returns a :class:`.PyTorchMeasureResult`

Variance-Based Invariance Measures
------------------------
* :class:`.SampleVarianceInvariance`
* :class:`.TransformationVarianceInvariance`
* :class:`.NormalizedVarianceInvariance`

Distance-Based Invariance Measures
------------------------
* :class:`.SampleDistanceInvariance`
* :class:`.TransformationDistanceInvariance`
* :class:`.NormalizedDistanceInvariance`

Other Invariance Measures
------------------------
* :class:`.GoodfellowInvariance`
* :class:`.ANOVAInvariance`
* :class:`.ANOVAFInvariance`

Variance-Based Same-Equivariance Measures
------------------------
* :class:`.SampleVarianceSameEquivariance`
* :class:`.TransformationVarianceSameEquivariance`
* :class:`.NormalizedVarianceSameEquivariance`

Distance-Based Same-Equivariance Measures
------------------------
* :class:`.SampleDistanceSameEquivariance`
* :class:`.TransformationDistanceSameEquivariance`
* :class:`.NormalizedDistanceSameEquivariance`

"""


from .base import NumpyMeasure
from .layer_transformation import MeasureTransformation, AggregateConvolutions, AggregateTransformation, \
    AggregateFunction, IdentityTransformation
from .quotient import QuotientMeasure, divide_activations

from .variance_invariance import NormalizedVarianceInvariance, SampleVarianceInvariance, \
    TransformationVarianceInvariance
from .anova import ANOVAInvariance, ANOVAFInvariance
from .distance_invariance import NormalizedDistanceInvariance, SampleDistanceInvariance, \
    TransformationDistanceInvariance, DistanceAggregation

from .distance_same_equivariance import NormalizedDistanceSameEquivariance, TransformationDistanceSameEquivariance, \
    SampleDistanceSameEquivariance

from .variance_same_equivariance import NormalizedVarianceSameEquivariance, TransformationVarianceSameEquivariance, \
    SampleVarianceSameEquivariance

from .same_equivariance_simple import DistanceSameEquivarianceSimple, DistanceFunction

from .goodfellow import GoodfellowInvariance
from .goodfellow_prob import GoodfellowNormalInvariance, GoodfellowNormalLocalInvariance, \
    GoodfellowNormalGlobalInvariance
