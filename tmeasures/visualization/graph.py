from __future__ import annotations

from typing import Any, Callable,TypeVar,TypeAlias
import typing
import torch

from tmeasures.utils import Graph,graph_str
from ..pytorch.model import named_children_deep

import graphviz as gv

NodeMaker = Callable[[gv.Digraph,torch.nn.Module,str,str],None]

def default_node_maker(g:gv.Digraph,module:torch.nn.Module,id:str,label:str):
    g.node(id,label=label)

def graph_to_dot_collect(t:Graph[torch.nn.Module], g:gv.Digraph,id:str,name:str,align_node:str,separator:str,make_node:NodeMaker)->tuple[str,str]:
    if isinstance(t,dict):
        cluster_name = f'cluster_{id}'
        align_node_new = f"al_{id}"
        first,last="",""
        with g.subgraph(name=cluster_name,graph_attr={"label":id}) as sgc:
            with sgc.subgraph(name=id) as sg:
                # sg.node(align_node_new,style="invis",height="0",width="0",peripheries="0",rank="0")
                # sg.edge(align_node,align_node_new,style="invis")
                sg.attr(rank='same')
                for k,v in t.items():
                    child_id = id+separator+k
                    child_first,child_last = graph_to_dot_collect(v,sg,child_id,k,align_node_new,separator,make_node)
                    if first =="":
                        first = child_first
                        sg.edge(align_node,child_first)
                    else:
                       sg.edge(last,child_id)
                    last = child_last

        return first,last
    else:
        assert isinstance(t,torch.nn.Module)
        make_node(g,t,id,name)
        return id,id
    
# def graph_to_dot_collect2(t:Graph[torch.nn.Module], g:gv.Digraph,prefix:str,align_node:str,separator:str,make_node:NodeMaker)->tuple[str,str]:
#   assert isinstance(t,dict)
#   last = ""
#   first = ""
#   for k,v in t.items():
#     unique_k = prefix+separator+k

#     if isinstance(v,dict):
#       cluster_name = f'cluster_{unique_k}'
#       align_node_new = f"al_{unique_k}"
#       with g.subgraph(name=cluster_name,graph_attr={"label":k}) as sgc:
#           with sgc.subgraph(name=k) as sg:
#             #sg.node(align_node_new,style="invis",height="0",width="0",peripheries="0",rank="0")
#             sg.attr(rank='same')
#             sg_first,sg_last = graph_to_dot_collect(v,sg,unique_k,align_node_new,separator,make_node)
#       if first =="":
#         first = sg_first
#         g.edge(align_node,first,style="invis")
#       if last != "":
#         g.edge(last,sg_first,constraints="false")
#       last = sg_last
#     else:
#       make_node(g,v,unique_k,k)
#       if first =="":
#         first = unique_k
#         g.edge(align_node,first,style="invis")
#       if last != "":
#         g.edge(last,unique_k)
#       last = unique_k
#   return first,last


def graph_to_dot(t:Graph[torch.nn.Module],make_node:NodeMaker=default_node_maker)->gv.Digraph:
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
  g.attr(color="blue")
  with g.subgraph(name="align") as sgc:
    sgc.node("al",label="Input",rank="0")
    first,last = graph_to_dot_collect(t,sgc,"network","network","al",".",make_node)
    sgc.edge("al",first)
  
  return g


