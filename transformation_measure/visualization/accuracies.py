import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from experiments.language import l

from . import default_discrete_colormap

def plot_accuracies(plot_filepath:Path, accuracies_by_label:[[float]], labels:[str], group_names:[str]):
    # set width of bar
    f=plt.figure(dpi=300)
    patterns = ["...","**","\\\\\\","///",  "+" , "x", "o", "O", ".", "*" ,"/" , "\\" , "|" , "-" ,]

    accuracies_by_label= accuracies_by_label.T
    n_groups=len(group_names)
    n_labels=len(labels)
    barWidth = 1/(n_labels+1)
    cmap = default_discrete_colormap()
    # Set position of bar on X axis
    pos = np.arange(n_groups,dtype=float)
    pad = barWidth*0.1
    for i,(accuracies,label) in enumerate(zip(accuracies_by_label, labels)):
        if n_labels <= len(patterns):
            hatch=patterns[i]
        else:
            hatch=None
    # Make the plot
        plt.bar(pos, accuracies, color=cmap(i), width=barWidth, edgecolor='white', label=label,hatch=hatch)
        pos+=barWidth+pad
    plt.gca().set_ylim(0,1)

    plt.gca().yaxis.grid(which="major", color='gray', linestyle='-', linewidth=0.5)
    # Add xticks on the middle of the group bars
    plt.xlabel(l.model)
    plt.ylabel(l.accuracy)
    def shorten(l:str): return l if len(l)<=10  else l[:9]+"."

    group_names = [shorten(l) for l in group_names]

    plt.xticks([r + barWidth for r in range(len(group_names))], group_names)
    plt.tick_params(axis='both', which='both', length=0)
    #plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='on')

    # Create legend & save
    plt.legend(fontsize=8)
    plt.savefig(plot_filepath,bbox_inches='tight')
    plt.close()

def plot_accuracies_single_model(plot_filepath:Path, accuracies:[float], labels:[str]):
    # set width of bar
    f=plt.figure(dpi=300)
    n=len(labels)
    assert len(accuracies)==n, f"Different number of labels {n} and accuracies {len(accuracies)} "
    cmap = default_discrete_colormap()
    # Set position of bar on X axis
    x = np.arange(n,dtype=float)
    colors = np.array([cmap(i) for i in range(n)])
    # print(colors)
    # Make the plot
    plt.bar(x, accuracies,color=colors, edgecolor='white')

    plt.gca().set_ylim(0,1)

    plt.gca().yaxis.grid(which="major", color='gray', linestyle='-', linewidth=0.5)
    # Add xticks on the middle of the group bars
    plt.xlabel(l.transformation)
    plt.ylabel(l.accuracy)

    plt.xticks(x, labels)
    plt.tick_params(axis='both', which='both', length=0)
    #plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='on')

    # Create legend & save
    plt.legend(fontsize=8)
    plt.savefig(plot_filepath,bbox_inches='tight')
    plt.close()


