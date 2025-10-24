import torch
from torch import nn
import torchvision

from tests.pytorch.fixtures import cifar10_resnet_variance_invariance_fixture, evaluate_fixture_default
from tmeasures.pytorch.model import named_children_deep
from tmeasures.visualization.graph import MeasureAverageVisualization, graph_to_dot

def test_graph_to_dot():
    #model = torchvision.models.AlexNet()
    model = torchvision.models.resnet50()
    g = named_children_deep(model)
    dot = graph_to_dot(g)
    dot.render("tests/pytorch/test",format="png")
    dot.save("tests/pytorch/test.dot")


def test_measure_to_dot():
    f = cifar10_resnet_variance_invariance_fixture()
    measure = evaluate_fixture_default(f)
    model = f.model(torch.device("cpu"),f.dataset)
    g = named_children_deep(model)
    node_maker = MeasureAverageVisualization(measure)
    dot = graph_to_dot(g,node_maker = node_maker)
    dot.render("tests/pytorch/test",format="png")
    dot.save("tests/pytorch/test.dot")