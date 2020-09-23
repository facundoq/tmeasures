# TODO move to experiments?
import transformational_measures as tm

params = {
    # 'text.latex.preamble': ['\\usepackage{gensymb}'],

    'font.family': ['computer modern','cm','serif'],
    # 'image.origin': 'lower',
    # 'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    # 'savefig.dpi': 150,  # to adjust notebook inline plot size
    'axes.labelsize': 14, # fontsize for x and y labels (was 10)
    'axes.titlesize': 8,
    'font.size': 8, # was 10
    'legend.fontsize': 10, # was 10
    'xtick.labelsize': 12,
    'ytick.labelsize': 10,
    # 'text.usetex': True,
    # 'figure.figsize': [3.39, 2.10],
    'axes.spines.bottom': False,
    'axes.spines.left': False,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.axisbelow' :  True,
    'axes.grid'      : True,
    'axes.grid.axis' : "y", #or "both" or "x" or "none"
    #'axes.grid.color' : "gray",
    #'axes.grid.linewidth' : 0.5,
}



import matplotlib
matplotlib.rcParams.update(params)
import matplotlib.pyplot as plt
plt.rc('grid', c='0.5', ls='-', lw=0.5)
# plt.rc('font', **{'family': 'cmu', 'serif': ['cm']})
plt.rc('text', usetex=False)

import numpy as np
from typing import List
from pathlib import Path
from transformational_measures.numpy.stats_running import  RunningMeanAndVarianceWelford
import transformational_measures as tm
from matplotlib.lines import Line2D

from ..language import l

default_y_lim=1.4

def plot_collapsing_layers_different_models(results:List[tm.MeasureResult], filepath:Path, labels=None, title="", linestyles=None, colors=None, legend_location=None, markers:[[int]]=None, ylim=None):
    if ylim is None:
        ylim = default_y_lim
    f=plt.figure(dpi=300)
    n = len(results)
    if n == 0:
        raise ValueError(f"`results` is an empty list.")
    colors = get_colors(colors, n)

    f, ax = plt.subplots(dpi=get_dpi(n))
    f.suptitle(title)

    if linestyles is None and n<=4:
        linestyles=["-","--",":","-."]

    result_layers = np.array([len(r.layer_names) for r in results])
    min_n, max_n = result_layers.min(), result_layers.max()
    max_value=0
    # x_result_most_layers=np.zeros(1)
    for i, result in enumerate(results):
        n_layers = len(result.layers)
        x = np.linspace(0,100,n_layers,endpoint=True)
        # if n_layers>=x_result_most_layers.size:
        #     x_result_most_layers=x
        y = result.per_layer_average()
        max_value = max(max_value, y.max())
        if labels is None:
            label = None
        else:
            label = labels[i]
        linestyle = get_default(linestyles,i,"-")

        color=colors[i, :]
        ax.plot(x, y, label=label, linestyle=linestyle, color=color,marker="o",markersize=3)

        if not markers is None:
            mark_layers = markers[i]
            x_mark = x[mark_layers]
            y_mark = y[mark_layers]
            ax.plot(x_mark, y_mark, linestyle="", color=color, marker="s")

    ax.set_ylabel(l.measure)
    ax.set_xlabel(f"{l.layer} (%)")
    ax.set_ylim(0, max(max_value * 1.1, ylim))

    ticks = list(range(0, 110, 10,))
    ax.set_xticks(ticks)


    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])


    if not labels is None:
        # Put legend below current axis
        handles, labels = ax.get_legend_handles_labels()
        handles_new=[Line2D([0],[0]) for h in handles]

        for h,h_new in zip(handles,handles_new):
            h_new.update_from(h)
            h_new.set_marker("")

        if legend_location is None:
            # loc, pos = ['lower center', np.array((0.5, 0))]
            ax.legend(handles_new,labels,fancybox=True, )
        else:
            loc, pos = legend_location
            ax.legend(handles_new,labels,loc=loc, bbox_to_anchor=pos,
                      fancybox=True)

    plt.savefig(filepath, bbox_inches="tight")
    plt.close()

def get_default(l:[],i:int,default):
    if l is None:
        return default
    else:
        return l[i]


def add_legends(ax, original_labels:[str], plot_mean:bool, legend_location):
    put_legend=False
    if plot_mean:
        handles, labels = ax.get_legend_handles_labels()
        handles_new =[handles[-1]]
        labels=[labels[-1]]
        put_legend=True
    else:
        if not original_labels is None:
            # Put legend below current axis
            handles, labels = ax.get_legend_handles_labels()
            handles_new = [Line2D([0], [0]) for h in handles]

            for h, h_new in zip(handles, handles_new):
                h_new.update_from(h)
                h_new.set_marker("")
            put_legend = True

    if put_legend:
        if legend_location is None:
            # loc, pos = ['lower center', np.array((0.5, 0))]
            ax.legend(handles_new, labels, fancybox=True)
        else:
            loc, pos = legend_location
            ax.legend(handles_new, labels, loc=loc, bbox_to_anchor=pos,
                      fancybox=True)

def shorten_layer_names(labels:[str])->[str]:
    result=[]
    for l in labels:
        i=0
        # get the index of the first letter after the last _
        chars=[str(n) for n in range(9)]+["_"]
        while i<len(l) and l[i] in chars: i+=1
        # remove all chars before the last _
        l=l[i:]

        # if l[1]=="_":
        #     l=l[2:]
        if l.startswith("fc"):
            l="lin"+l[2:]
        if l=="c":
            l="conv"
        if l.endswith("MaxPool2d"):
            l=l[:-9]+"mp"
            result.append(l)
        elif l.endswith("Flatten"):
            l=l[:-7]+"vect"
            result.append(l)
        else:
            result.append(l)
    return result

def get_colors(colors:np.ndarray,n:int)->np.ndarray:
    if colors is None:
        if n==2:
            colors = np.array([[1, 0, 0], [0, 0, 1]])
        else:
            colors = plt.cm.rainbow(np.linspace(0.01, 1, n))
    return colors

def get_dpi(n:int):
    return min(350, max(150, n * 15))


def plot_collapsing_layers_same_model(results:List[tm.MeasureResult], filepath:Path, labels:[str]=None, title="", linestyles=None, plot_mean=False, colors=None, legend_location=None, mark_layers:[int]=None, ylim=None):
    if ylim is None:
        ylim = default_y_lim

    n=len(results)
    if n == 0:
        raise ValueError(f"`results` is an empty list.")
    colors=get_colors(colors,n)

    f, ax = plt.subplots(dpi=get_dpi(n))
    f.suptitle(title)

    result_layers=np.array([len(r.layer_names) for r in results])
    min_n,max_n = result_layers.min(),result_layers.max()
    if plot_mean:
        assert min_n==max_n,"To plot the mean values all results must have the same number of layers."

    if linestyles is None and n <= 4 and plot_mean == False:
        linestyles = ["-", "--", ":", "-."]

    mean_and_variance = RunningMeanAndVarianceWelford()
    max_value=0
    for i, result in enumerate(results):
        n_layers= len(result.layers)
        x= np.arange(n_layers)+1
        y= result.per_layer_average()
        max_value = max(max_value,y.max())
        if plot_mean:
            mean_and_variance.update(y)
        label = get_default(labels,i,None)
        linestyle = get_default(linestyles,i,"-")
        color=colors[i, :]
        # if plot_mean:
        #     color*=0.7

        ax.plot(x, y, label=label, linestyle=linestyle, color=color,marker="o",markersize=3)

        if not mark_layers is None:
            x_mark = x[mark_layers]
            y_mark = y[mark_layers]
            ax.plot(x_mark,y_mark,linestyle="",color=color,marker="s")
    if plot_mean:
        x = np.arange(max_n)+1
        y,error=mean_and_variance.mean(),mean_and_variance.std()
        label=l.meandeviation
        linestyle="--"
        ax.errorbar(x, y,yerr=error, label=label, linewidth=1.5, linestyle=linestyle, color=(0,0,0))
    else:
        handles, _ = ax.get_legend_handles_labels()

    ax.set_ylabel(l.measure)
    ax.set_xlabel(l.layer)
    ax.set_ylim(0,max(max_value*1.1,ylim))

    if max_n < 60:
        tick_labels = results[0].layer_names
        tick_labels = shorten_layer_names(tick_labels)
        #labels = [f"${l}$" for l in labels]
        x = np.arange(max_n) + 1
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, rotation=90)
        ax.tick_params(axis='both', which='both', length=0)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    add_legends(ax,labels,plot_mean,legend_location)
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()




