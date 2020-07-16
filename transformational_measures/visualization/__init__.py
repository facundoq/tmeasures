import matplotlib
import matplotlib.pyplot as plt
from transformational_measures.numpy.base import NumpyMeasure
from .. import MeasureResult
from ..language import l

def discrete_colormap(n:int=16,base_colormap="rainbow",):
    colors = plt.cm.get_cmap(base_colormap, n)(range(n))
    cm = matplotlib.colors.ListedColormap(colors)
    return cm


def default_discrete_colormap():
    return plt.cm.get_cmap("Set1")

# def labels_for_measure_results(ms:[MeasureResult]):
#     return [label_for_measure_result(m) for m in ms]
# def label_for_measure_result(m:MeasureResult):
#     return label_for_measure(m.measure)
#
#
# def label_for_measure(m:NumpyMeasure):
#     return l.measure_name(m)
#
# def labels_for_measures(ms:[MeasureResult]):
#     return [label_for_measure(m) for m in ms]

def get_sequential_colors(values):
    cmap= plt.cm.get_cmap("plasma",len(values))
    colors = cmap(values)
    return colors

from .layers import plot_collapsing_layers_same_model,plot_collapsing_layers_different_models
from .heatmaps import plot_heatmap
from .features import plot_invariant_feature_maps




