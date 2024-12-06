from sklearn.decomposition import NMF
import torch
import numpy as np

from tmeasures.visualization.images import plot_images_multichannel,plot_images_rgb

def reorder_conv2d_weights(activation:torch.nn.Module,invariance:np.array):
    with torch.no_grad():
        weight = dict(activation.named_parameters())["weight"]
        indices = invariance.argsort().copy()
        weight[:] = weight[indices,:,:,:]
        invariance[:] = invariance[indices]


def sort_weights_invariance(weights:np.array,invariance:np.array,top_k:int=None):
    indices = invariance.argsort()
    if not top_k is None:
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
    if not max_inputs is None:
        weights[:,:max_inputs,]
    return weights

def plot_conv2d_filters(conv2d:torch.nn.Module,invariance:np.array,sort=True, top_k=None,max_inputs=10,nmf_components=None):
    weights = dict(conv2d.named_parameters())["weight"].detach().numpy()
    mi,ma=weights.min(),weights.max()

    if sort or not top_k is None :
        weights, invariance = sort_weights_invariance(weights,invariance,top_k)
    if not nmf_components is None:
        weights = weights_reduce_nmf(weights,nmf_components)
    if not max_inputs is None:   
        weights = weight_inputs_filter_importance(weights,max_inputs)
    largest = max(abs(mi),abs(ma))
    vmin,vmax = -largest,largest
    # print(weights.shape)
    plot_images_multichannel(weights,invariance,vmin,vmax)

def plot_conv2d_filters_rgb(conv2d:torch.nn.Module,invariance:np.array):
    weights = dict(conv2d.named_parameters())["weight"].detach().numpy()
    weights, invariance = sort_weights_invariance(weights,invariance)
    weights = weights_reduce_nmf(weights,3)

    mi,ma=weights.min(),weights.max()
    largest = max(abs(mi),abs(ma))
    vmin,vmax = -largest,largest
    plot_images_rgb(weights,invariance,vmin,vmax)