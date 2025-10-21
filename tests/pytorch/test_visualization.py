import torch
from torch import nn
import torchvision

from tmeasures.pytorch.model import named_children_deep
from tmeasures.visualization.graph import graph_to_dot

def test_graph_to_dot():
    model = torchvision.models.AlexNet()
    #model = torchvision.models.resnet50()
    g = named_children_deep(model)
    dot = graph_to_dot(g)
    dot.render("tests/pytorch/test.png",format="png")
    dot.save("tests/pytorch/test.dot")
