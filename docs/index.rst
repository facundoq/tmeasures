.. tmeasures documentation master file, created by
   sphinx-quickstart on Thu Jul  7 06:23:57 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

✴ Transformational Measures 📏
=====================================


.. meta::
    :property=og\:image: https://raw.githubusercontent.com/facundoq/tmeasures/master/docs/img/probflow_og_image.png
    :property=og\:image\:width: 1200
    :property=og\:image\:height: 1200

|Version Badge|  |Build Badge|  |Docs Badge|  |Coverage Badge| |Colab Badge|

.. |Version Badge| image:: https://img.shields.io/pypi/v/tmeasures
    :target: https://pypi.org/project/tmeasures/

.. |Build Badge| image:: https://github.com/facundoq/tmeasures/workflows/tests/badge.svg
    :target: https://github.com/facundoq/tmeasures/actions?query=branch%3Amaster

.. |Docs Badge| image:: https://readthedocs.org/projects/probflow/badge/
    :target: http://tmeasures.readthedocs.io

.. |Coverage Badge| image:: https://codecov.io/gh/facundoq/tmeasures/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/facundoq/tmeasures

.. |Colab Badge| image:: https://github.com/facundoq/tmeasures/docs/res/colab-badge.svg
    :target: https://colab.research.google.com/github/facundoq/transformational_measures/blob/master/docs/examples/Variance%20to%20rotations%20of%20a%20CNN%20trained%20on%20MNIST%20with%20PyTorch.ipynb

🔎 Visualizations
---------------------_

``tmeasures`` allows computing invariance, same-equivariance and other
transformational measures, and contains helpful functions to visualize
these. The following are some examples of the results you can obtain
with the library:

🔥 Invariance heatmap
~~~~~~~~~~~~~~~~~~~~~~~

Each column shows the invariance to rotation of a layer of a Neural
Network. Each row/block inside each column indicates the invariance of a
feature map or single neuron, depending on the layer.

|image0|

📉 Average Invariance vs layer, same model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plot the transformational and sample invariance to rotations of a simple
neural network trained on MNIST, with and without data augmentation. The
X axis indicates the layer, while the Y axis shows the average
invariance of the layer.

|image1|

📈 Average invariance by layer, different models:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plot of the invariance to rotations of several well-known models trained
on CIFAR10. The number of layers of each model is streched on a
percentage scale, so that different models can be compared.

|image2|

💻 PyTorch API
---------------

The following notebook contains a step-by-step to measure invariance to
rotations in a PyTorch neural network that was trained with the MNIST
dataset. You can execute it directly from google colab.

`Measuring invariance to rotations for a simple CNN on MNIST (google
colab) <https://colab.research.google.com/github/facundoq/transformational_measures/blob/master/docs/examples/Variance%20to%20rotations%20of%20a%20CNN%20trained%20on%20MNIST%20with%20PyTorch.ipynb>`__

Other examples with multiple measures and pretrained models can be found
in the `doc </doc>`__ folder of this repository.

💻 TensorFlow API
------------------

We are still developing the Tensorflow API.

📋 Examples
------------

You can find many uses of this library in the `repository with the
code <https://github.com/facundoq/transformational_measures_experiments>`__
for the article `Measuring (in)variances in Convolutional
Networks <https://link.springer.com/chapter/10.1007/978-3-030-27713-0_9>`__,
where this library was first presented. Also, in the code for the
experiments of the PhD Thesis `“Invariance and Same-Equivariance
Measures for Convolutional Neural Networks”
(spanish) <https://doi.org/10.24215/16666038.20.e06>`__.

🤙🏽 Citing
------------

If you use this library in your research, we kindly ask you to cite
`Invariance and Same-Equivariance Measures for Convolutional Neural
Networks. <https://doi.org/10.24215/16666038.20.e06>`__

::

   @article{quiroga20,
     author    = {Facundo Quiroga and
                  Laura Lanzarini},
     title     = {Invariance and Same-Equivariance Measures for Convolutional Neural Networks},
     journal   = {J. Comput. Sci. Technol.},
     volume    = {20},
     number    = {1},
     pages     = {06},
     year      = {2020},
     url       = {https://doi.org/10.24215/16666038.20.e06},
     doi       = {10.24215/16666038.20.e06},
   }

.. |image0| image:: ../../res/heatmap.png
.. |image1| image:: res/invariance.jpg
.. |image2| image:: res/invariance_models_cifar10.jpg




.. toctree::
   :maxdepth: 4
   :caption: Contents:
   :hidden:

   source/modules
   development



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
