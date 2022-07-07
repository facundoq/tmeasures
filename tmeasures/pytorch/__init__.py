
"""
The :mod:`.pytorch` module contains implementations of the measures for the PyTorch framework, as well as some utility functions.

A :class:`.PyTorchMeasure` can be evaluated with a PyTorch :class:`.ActivationsModule`, dataset and :class:`.PyTorchTransformationSet`. The evaluation of the measure returns a :class:`.PyTorchMeasureResult`

Variance-Based Invariance Measures
------------------------
* :class:`.SampleVarianceInvariance`
* :class:`.TransformationVarianceInvariance`
* :class:`.NormalizedVarianceInvariance`

Other Invariance Measures
------------------------
* :class:`.GoodfellowInvariance`

Variance-Based Same-Equivariance Measures
------------------------
* :class:`.SampleVarianceSameEquivariance`
* :class:`.TransformationVarianceSameEquivariance`
* :class:`.NormalizedVarianceSameEquivariance`
"""

from .model import ActivationsModule,AutoActivationsModule
from .base import PyTorchMeasure,PyTorchMeasureOptions,ActivationsByLayer,PyTorchMeasureResult
from .transformations import PyTorchTransformationSet,PyTorchTransformation

from .activations_iterator_base import PytorchActivationsIterator,InvertedPytorchActivationsIterator,BothPytorchActivationsIterator,NormalPytorchActivationsIterator
from .measure_transformer import MeasureTransformation,NoTransformation,AverageFeatureMaps
from .util import SequentialWithIntermediates
from . import dataset2d

from .layer_measures import Variance

from .variance_invariance import SampleVarianceInvariance,TransformationVarianceInvariance,NormalizedVarianceInvariance
from .variance_sameequivariance import  SampleVarianceSameEquivariance,TransformationVarianceSameEquivariance, NormalizedVarianceSameEquivariance
from .goodfellow import GoodfellowInvariance,NormalPValueThreshold,PercentActivationThreshold