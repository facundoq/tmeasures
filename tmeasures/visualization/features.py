import numpy as np
import matplotlib.pyplot as plt
import tmeasures as tm
from pathlib import Path
from ..np import AggregateTransformation,AggregateFunction,MeasureTransformation
from .. import MeasureResult
import torch

from tmeasures.np.activations_iterator import ActivationsIterator

def plot_activations(features:np.ndarray, feature_indices:[int], variance_scores:[float], x_transformed:np.ndarray, transformations:tm.TransformationSet, filepath:Path):

    x_transformed= x_transformed.transpose((0, 2, 3, 1))
    n,c = features.shape
    nt,ht,wt,ct = x_transformed.shape

    dpi = int(np.sqrt(n) * 9) + 150
    f, axis = plt.subplots(c+1,n+1,dpi=dpi)
    axis[0, 0].set_title("Inputs")
    axis[0, 0].axis("off")

    for i in range(c):
        axis[i+1, 0].set_title(f"Act {feature_indices[i]} \n, score:{variance_scores[i]:0.2f} ",fontsize=5)
        axis[i+1, 0].axis("off")

    for j in range(n):
        transformed_image= x_transformed[j, :]
        transformed_image-=transformed_image.min()
        transformed_image/= transformed_image.max()
        if ct ==1:
            transformed_image=transformed_image[:,:,0]
        axis[0,j+1].imshow(transformed_image,cmap="gray")
        axis[0,j+1].axis("off")

    for i in range(c):
        mi, ma = features[:,i].min(), features[:,i].max()
        for j in range(n):
            image = features[j, i].reshape((1,1))
            axis[i+1,j+1].imshow(image, vmin=mi, vmax=ma,cmap="gray")
            axis[i+1, j+1].axis("off")

    plt.savefig(filepath)
    plt.close("all")

def plot_feature_maps(feature_maps:np.ndarray, feature_indices:[int], variance_scores:[float], x_transformed:np.ndarray, transformations:tm.TransformationSet, filepath:Path):
    feature_maps = feature_maps.transpose((0,2,3,1))
    x_transformed= x_transformed.transpose((0, 2, 3, 1))
    n,h,w,c = feature_maps.shape
    nt,ht,wt,ct = x_transformed.shape
    if ct == 1:
        x_transformed = x_transformed[:,:, :, 0]

    fontsize=max(3, 10 - int(np.sqrt(n)))
    dpi = int(np.sqrt(n) * 9) + 100
    f, axis = plt.subplots(c+1, n+4, dpi=dpi)
    axis[0, 0].set_title("Inputs")
    axis[0, 0].axis("off")
    for i in range(c):
        title=f"FM {feature_indices[i]}\n, score:\n{variance_scores[i]:0.2f} "
        axis[i+1, 0].text(0,0, title,fontsize=fontsize)
        axis[i+1, 0].axis("off")

    for j in range(n):
        transformed_image  = x_transformed[j, :]
        transformed_image -= transformed_image.min()
        transformed_image /= transformed_image.max()

        axis[0,j+1].imshow(transformed_image)
        axis[0,j+1].axis("off")

    colorbar_images=[]
    for i in range(c):
        mi, ma = feature_maps[:,:,:,i].min(), feature_maps[:,:,:,i].max()
        for j in range(n):
            im=axis[i+1,j+1].imshow(feature_maps[j,:,:,i],vmin=mi,vmax=ma,cmap="gray")
            axis[i+1, j+1].axis("off")
            if j+1 == n:
                colorbar_images.append(im)

    # mean and std of feature maps columns
    for i in range(c):
        mean_feature_map = np.mean(feature_maps[:,:,:,i],axis=0)
        std_feature_map = np.std(feature_maps[:, :, :, i], axis=0)
        axis[i + 1, -3].imshow(mean_feature_map,cmap="gray")
        axis[i + 1, -3].axis("off")
        axis[i + 1, -2].imshow(std_feature_map,cmap="gray")
        axis[i + 1, -2].axis("off")

    for i in range(c):
        axis[i + 1, -1].axis("off")
        cbar = plt.colorbar(colorbar_images[i], ax=axis[i + 1, -1])
        cbar.ax.tick_params(labelsize=fontsize)
    axis[0, -1].axis("off")

        # mean and std of images columns
    axis[0, -3].imshow(x_transformed.mean(axis=0),cmap="gray")
    axis[0, -3].axis("off")

    axis[0, -2].imshow(x_transformed.std(axis=0),cmap="gray")
    axis[0, -2].axis("off")

    plt.savefig(filepath)
    plt.close("all")

def indices_of_smallest_k(a,k):

    indices = np.argpartition(a, k)[:k]
    values = a[indices]
    indices = np.argsort(a)

    # ind.sort()
    return indices[:k]

def indices_of_largest_k(a,k):

    # indices = np.argpartition(a, -k)[:k]
    # values = a[indices]
    indices=np.argsort(a)

    # ind.sort()
    return indices[-k:]

def select_feature_maps(measure_result: tm.MeasureResult, most_invariant_k:int, least_invariant_k:int):
    feature_indices_per_layer=[]
    feature_scores_per_layer = []
    values=measure_result.layers
    layer_names=measure_result.layer_names
    for value,name in zip(values,layer_names):
        if len(value.shape) != 1:
            raise ValueError("Feature maps should be collapsed before calling this function")
        most_invariant_indices = indices_of_smallest_k(value, most_invariant_k)
        least_invariant_indices = indices_of_largest_k(value, least_invariant_k)
        indices = np.concatenate([most_invariant_indices,least_invariant_indices])
        feature_indices_per_layer.append(indices)
        feature_scores_per_layer.append(value[indices])
    return feature_indices_per_layer,feature_scores_per_layer

'''
    Plots the activation of the invariant feature maps determined by :param result
    Plots the best :param features_per_layer feature maps, ie the most invariant
    Creates a plot for each sample image/transformation pair
    '''

from tmeasures.np import MeasureTransformation

def plot_invariant_feature_maps(plot_folderpath:Path, activations_iterator:ActivationsIterator, result:tm.MeasureResult, most_invariant_k:int, least_invariant_k:int
                                #, conv_aggregation:MeasureTransformation
                                ):
    #result=conv_aggregation.apply(result)

    feature_indices_per_layer,invariance_scores_per_layer=select_feature_maps(result, most_invariant_k,least_invariant_k)
    transformations=activations_iterator.get_transformations()

    for i_image,(x,transformation_activations_iterator) in enumerate(activations_iterator.samples_first()):
        layers_activations,x_transformed = activations_iterator.row_from_iterator(transformation_activations_iterator)
        for i_layer,layer_activations in enumerate(layers_activations):
            layer_name=result.layer_names[i_layer]
            filepath = plot_folderpath / f"image{i_image}_layer{i_layer}_{layer_name}.png"
            feature_indices=feature_indices_per_layer[i_layer]
            invariance_scores=invariance_scores_per_layer[i_layer]
            #only plot conv layer activations
            if len(layer_activations.shape)==4:
                feature_maps=layer_activations[:,feature_indices,:,:]
                plot_feature_maps(feature_maps,feature_indices,invariance_scores,x_transformed,transformations,filepath)
            elif len(layer_activations.shape)==2:
                features = layer_activations[:, feature_indices]
                plot_activations(features, feature_indices, invariance_scores, x_transformed, transformations,filepath)




