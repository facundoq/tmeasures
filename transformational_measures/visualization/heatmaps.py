import numpy as np
import matplotlib.pyplot as plt
from transformational_measures import MeasureResult
import transformational_measures as tm
from pathlib import Path



def get_limit(m:MeasureResult, op_code:str):
    ops={"max":np.nanmax,"min":np.nanmin}
    op = ops[op_code]
    # TODO ignore array if its all inf
    vals = [l[l!=np.inf] for l in m.layers]
    vals = [l for l in vals if len(l)>0]
    vals = np.array([op(l) for l in vals])
    return op(vals)


def plot_heatmap(m:MeasureResult,filepath:Path, vmin=None, vmax=None):
    for l in m.layers:
        print(l.shape, "inf: ", np.sum(np.isinf(l)),", nan: ",np.sum(np.isnan(l)),"min/mean/max:",l.min(),l.mean(),l.max())
    #
    # print("Stats for: ",m.measure.id())
    # for l,n in zip(m.layers,m.layer_names):
    #     print(f"{n}: ",l.shape, np.sum(np.isinf(l)),"min/mean/max:",l.min(),l.mean(),l.max())
    for i,l in enumerate(m.layers):
        d=len(l.shape)
        if d>1:
            dims = tuple(range(1,d))
            m.layers[i]=np.nanmean(l,axis=dims)

    # m=tm.numpy.AggregateTransformation(tm.numpy.AggregateFunction.mean,[0]).apply(m)

    print("Post normalization:")
    for l, n in zip(m.layers, m.layer_names):
        print(f"{n}: ",l.shape, np.sum(np.isinf(l)),"min/mean/max:",l.min(),l.mean(),l.max())



    if vmax is None: vmax = get_limit(m, "max")
    if vmin is None:
        vmin = get_limit(m, "min")
        if vmin > 0:
            vmin = 0

    n = len(m.layer_names)

    f, axes = plt.subplots(1, n, dpi=150, squeeze=False)
    mappable=None
    for i, (activation, name) in enumerate(zip(m.layers, m.layer_names)):
        ax = axes[0,i]
        ax.axis("off")
        activation = activation[:, np.newaxis]
        # print(activation.shape, activation.min(), activation.max())
        mappable = ax.imshow(activation,vmin=vmin,vmax=vmax,cmap='inferno',aspect="auto")

        if n<40:
            if len(name)>7:
                name=name[:6]+"."
            ax.set_title(name, fontsize=4,rotation = 45)

        # logging.debug(f"plotting stats of layer {name} of class {class_id}, shape {stat.mean().shape}")
    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar=f.colorbar(mappable, cax=cbar_ax, extend='max')
    cbar.cmap.set_over('green')
    cbar.cmap.set_bad(color='gray')
    plt.savefig(filepath)
    plt.close()



