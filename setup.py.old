from setuptools import find_packages, setup, Command
from shutil import rmtree
import sys
import os
import io
from pathlib import Path

from os import path
this_directory = os.path.abspath(os.path.dirname(__file__))

with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


url="https://github.com/facundoq/tmeasures"
VERSION='1.2.9'

setup(
    name="tmeasures",
    version=VERSION,
    python_requires='>=3.6',
    packages=find_packages(),
    scripts=[],

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "data-science-types",
        "tqdm",
        "opencv-python",
        "scikit-image",
        "statsmodels",
    ],

    package_data={
        # If any package contains *.txt or *.rst files, include them:
        "": ["*.txt", "*.rst"],
        # And include any *.msg files found in the "hello" package, too:
        "hello": ["*.msg"],
    },
    zip_safe=True,
    # metadata to display on PyPI
    author="Facundo Manuel Quiroga",
    author_email="facundoq@gmail.com",
    description="Invariance, Same-Equivariance and other measures for Neural Networks. Support for PyTorch (now) and TensorFlow (coming).",
    keywords="transformational measures equivariance same-equivariance invariance variance neural networks python pytorch numpy tensorflow",
    url=url,   # project home page, if any
    project_urls={
        "Bug Tracker": url+"/issues",
        "Documentation": url,
        "Source Code": url,
    },
    # check list at:
    # https://pypi.org/classifiers/
    classifiers=[
        "License :: OSI Approved :: Python Software Foundation License",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Visualization",
        
    ],

    # could also include long_description, download_url, etc.
    long_description=long_description,
    long_description_content_type='text/markdown',
)
