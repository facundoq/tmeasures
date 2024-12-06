import numpy as np
import matplotlib.pyplot as plt

    
def plot_images_multichannel(images,invariance,vmin,vmax,colorbar_space=1):
    N,C,H,W = images.shape
    invariance_fontsize = np.sqrt(C)*2
    f, subplots = plt.subplots(N,C,dpi=150,figsize=(C+colorbar_space,N))
    for i in range(N):
        for j in range(C):
            filter_weights = images[i,j,:,:]
            ax = subplots[i,j]
            ax.set_axis_off()
            im = ax.imshow(filter_weights,cmap="PuOr",vmin=vmin,vmax=vmax,interpolation='nearest')
            label = f"{invariance[i]:.02}"
            ax.text(0.5, 0.5,label, horizontalalignment='center',verticalalignment='center', transform=ax.transAxes,fontsize=invariance_fontsize)
    right=colorbar_space/(C+colorbar_space)
    
    f.subplots_adjust(right=1-right)
    gap = 0.2
    cbar_ax = f.add_axes([1-right*(1-gap), 0.15, right*(1-2*gap), 0.7])
    f.colorbar(im, cax=cbar_ax)