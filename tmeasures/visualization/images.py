import numpy as np
import matplotlib.pyplot as plt

def plot_images_rgb(images,labels=None,cols=None):
    N,C,H,W = images.shape
    if cols is None:
        cols = np.floor(np.sqrt(N)).astype(int)
    rows = (N // cols) + (1 if N % cols >0 else 0)
    invariance_fontsize = np.sqrt(cols)*2
    f, subplots = plt.subplots(rows,cols,dpi=100,figsize=(cols,rows))
    for i in range(rows):
        for j in range(cols):
            ax = subplots[i,j]
            ax.set_axis_off()
            index = i*cols+j
            if index <N:
                filter_weights = images[index,]
                ax.imshow(filter_weights,cmap="PuOr",interpolation='nearest')
                if not labels is None:
                    ax.text(0.5, 0.5,labels[index], horizontalalignment='center',verticalalignment='center', transform=ax.transAxes,fontsize=invariance_fontsize,c="yellow")
    
def plot_images_multichannel(images,vmin,vmax,colorbar_space=1,labels=None):
    N,C,H,W = images.shape
    invariance_fontsize = np.sqrt(C)*2
    f, subplots = plt.subplots(N,C,dpi=150,figsize=(C+colorbar_space,N))
    for i in range(N):
        for j in range(C):
            filter_weights = images[i,j,:,:]
            ax = subplots[i,j]
            ax.set_axis_off()
            im = ax.imshow(filter_weights,cmap="PuOr",vmin=vmin,vmax=vmax,interpolation='nearest')
            if not labels is None:
                ax.text(0.5, 0.5,labels[i], horizontalalignment='center',verticalalignment='center', transform=ax.transAxes,fontsize=invariance_fontsize)
    right=colorbar_space/(C+colorbar_space)
    
    f.subplots_adjust(right=1-right)
    gap = 0.2
    cbar_ax = f.add_axes([1-right*(1-gap), 0.15, right*(1-2*gap), 0.7])
    f.colorbar(im, cax=cbar_ax)