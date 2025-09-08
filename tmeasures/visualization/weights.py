import numpy as np
import torch
from sklearn.decomposition import NMF

from tmeasures.visualization.images import plot_images_multichannel, plot_images_rgb


def reorder_conv2d_weights(activation:torch.nn.Module,invariance:np.array):
    with torch.no_grad():
        weight = dict(activation.named_parameters())["weight"]
        indices = invariance.argsort().copy()
        weight[:] = weight[indices,:,:,:]
        invariance[:] = invariance[indices]


def sort_weights_invariance(weights:np.array,invariance:np.array,top_k:int=None):
    indices = invariance.argsort()
    if top_k is not None:
        indices = np.concatenate( [indices[:top_k],indices[-top_k:]])
    weights = weights[indices,:,:,:]
    invariance = invariance[indices]
    return weights, invariance


def weights_reduce_nmf(weights:np.array,n_components:int):
    Fo,Fi,H,W = weights.shape
    model = NMF(n_components=n_components, init='random', random_state=0,max_iter=1000)
    nmf_weights = np.zeros((Fo,n_components,H,W))
    for i in range(Fo):
        input_weights = weights[i,]
        flattened_weights = np.abs(input_weights.reshape(Fi,-1))
        model.fit(flattened_weights)
        nmf_weights[i] = model.components_.reshape(n_components,H,W)
    return nmf_weights

def weight_inputs_filter_importance(weights:np.array,max_inputs:int):
    Fo,Fi,H,W = weights.shape
    input_importance =  weights.mean(axis=(2,3))
    for i in range(Fo):
        indices = np.argsort(input_importance[i,:])[::-1]
        weights[i,:,:,:] = weights[i,indices,:,:]
    if max_inputs is not None:
        weights[:,:max_inputs,]
    return weights

def plot_conv2d_filters(conv2d:torch.nn.Module,invariance:np.array,sort=True, top_k=None,max_inputs=10,nmf_components=None):
    weights = dict(conv2d.named_parameters())["weight"].detach().numpy()
    mi,ma=weights.min(),weights.max()

    if sort or top_k is not None :
        weights, invariance = sort_weights_invariance(weights,invariance,top_k)
    if nmf_components is not None:
        weights = weights_reduce_nmf(weights,nmf_components)
    if max_inputs is not None:
        weights = weight_inputs_filter_importance(weights,max_inputs)
    largest = max(abs(mi),abs(ma))
    vmin,vmax = -largest,largest
    # print(weights.shape)
    labels = [f"{i:.02}" for i in invariance]
    plot_images_multichannel(weights,vmin,vmax,labels=labels)

def plot_conv2d_filters_rgb(conv2d:torch.nn.Module,invariance:np.array):
    weights = dict(conv2d.named_parameters())["weight"].detach().numpy()
    weights, invariance = sort_weights_invariance(weights,invariance)

    magnitudes = np.abs(weights).mean(axis=(1,2,3))

    weights = weights_reduce_nmf(weights,3)
    labels = [f"{i:.02}" for i,m in zip(invariance,magnitudes)]
    plot_images_rgb(weights,labels=labels)
