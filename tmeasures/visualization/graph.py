from __future__ import annotations

from typing import Any, Callable,TypeVar,TypeAlias
import typing
import torch

from tmeasures.measure import MeasureResult
from tmeasures.utils import Graph,graph_str
from ..pytorch.model import named_children_deep

import graphviz as gv

NodeMaker = Callable[[gv.Digraph,torch.nn.Module,str,str,dict[str,Any]],None]

def default_node_maker(g:gv.Digraph,module:torch.nn.Module,id:str,label:str,attrs:dict[str,Any]):
    g.node(id,label=label,**attrs)

class MeasureAverageVisualization:
    def __init__(self,measure:MeasureResult):
        self.measure=measure
    def __apply__(self, g:gv.Digraph,module:torch.nn.Module,id:str,label:str,attrs:dict[str,Any]):
        g.node(id,label=label+f" {self.measure.layers_dict()[id].mean():.2f}",**attrs)




def graph_to_dot_collect(t:Graph[torch.nn.Module], g:gv.Digraph,id:str,name:str,last:str,separator:str,make_node:NodeMaker)->tuple[str,str]:
    if isinstance(t,dict):
        cluster_name = f'cluster_{id}'
        first = ""
        with g.subgraph(name=cluster_name,graph_attr={"label":name}) as sgc:
            with sgc.subgraph(name=id) as sg:
                for k,v in t.items():
                    child_id = id+separator+k
                    child_first,child_last = graph_to_dot_collect(v,sg,child_id,k,last,separator,make_node)
                    if first == "" and child_first != "":
                        first = child_first
                    print(child_id,child_first,child_last,first,last)
                    if last !="" and not isinstance(v,dict):
                        sg.edge(last,child_first)
                    last = child_last
                    
        return first,last
    else:
        assert isinstance(t,torch.nn.Module)
        make_node(g,t,id,name)
        return id,id
    
def graph_to_dot_collect_dict(t:Graph[torch.nn.Module], g:gv.Digraph,id:str,first_al:str,separator:str,make_node:NodeMaker)->tuple[str,str]:
  assert isinstance(t,dict)
  last = ""
  first = ""
  for k,v in t.items():
    unique_k = id+separator+k

    if isinstance(v,dict):
      cluster_name = f'cluster_{unique_k}'
      with g.subgraph(name=cluster_name,graph_attr={"label":k}) as sgc:
          with sgc.subgraph(name=k) as sg:
            sg.attr(rank='same')
            sg_first,sg_last = graph_to_dot_collect_dict(v,sg,unique_k,first_al,separator,make_node)
      if first =="":
        first = sg_first
        # g.edge(first_al,first,style="invis")
      if last != "":
        g.edge(last,sg_first,constraints="false")
      last = sg_last
    else:
      make_node(g,v,unique_k,k,{"rank":"0"})
      if first =="":
        first = unique_k
        # g.edge(first_al,first,style="invis")
      if last != "":
        g.edge(last,unique_k)
      last = unique_k
  return first,last


def graph_to_dot(t:Graph[torch.nn.Module],node_maker:NodeMaker=default_node_maker)->gv.Digraph:
  attr = {"color":"blue",
          "rankdir":"TD",
          "splines":"ortho",
        }
  node_attr = {
      'shape': 'rect',
      'style':'rounded',
  }
  g = gv.Digraph(graph_attr=attr,node_attr=node_attr)
  g.attr(compound='true')
  g.attr(label='Network')
  with g.subgraph(name="align") as sgc:
    sgc.node("al",label="Input",rank="0")
    first,last = graph_to_dot_collect_dict(t,sgc,"network","al",".",node_maker)  
    sgc.edge("al",first)

  return g


