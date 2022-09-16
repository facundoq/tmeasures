import matplotlib
import matplotlib.pyplot as plt
from .. import MeasureResult


def discrete_colormap(n:int=16,base_colormap="rainbow",):
    colors = plt.cm.get_cmap(base_colormap, n)(range(n))
    cm = matplotlib.colors.ListedColormap(colors)
    return cm


def default_discrete_colormap():
    return plt.cm.get_cmap("Set1")

def get_sequential_colors(values):
    cmap= plt.cm.get_cmap("plasma",len(values))
    colors = cmap(values)
    return colors

from .layers import plot_average_activations_same_model,plot_average_activations_different_models,plot_average_activations,scatter_same_model
from .heatmaps import plot_heatmap
from .features import plot_invariant_feature_maps
from .sample_size import plot_relative_error_heatmap,get_relative_errors


