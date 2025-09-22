from __future__ import annotations

from typing import Any,TypeVar

def indices_of(l:list,value)->list[int]:
    return [i for i,l in enumerate(l) if value == l]    

T = TypeVar('T')

def get_all(l:list[T],indices:list[int])->list[T]:
    return [l[i] for i in indices]

from queue import Queue
from typing import Sized,Iterable

# class IterableQueue(Queue,Sized,Iterable):
#     """Queue supporting Iterator and Sized protocols.
#     Queue has a max size so it can be iterated upon with a for loop
#     """
#     def __init__(self, n:int,maxsize=None):
#         if maxsize is None:
#             maxsize=n
#         super().__init__(maxsize)
#         self.n=n
#         self.i=0

#     def __len__(self):
#         return self.n


#     def __iter__(self):
#         return self

#     def __next__(self):
#         self.i+=1
#         if self.i == self.n:
#             raise StopIteration()
#         return self.get()



def intersect_lists(a:list[T],b:list[T])->list[T]:
    return list(set(a) & set(b))

class DuplicateKeyError(ValueError):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

T = TypeVar('T')
type Graph[T] = dict[str,Graph[T]] | T

# Input = Union[MutableMapping,Any]
# TODO remove
# def flatten_dict(d_or_val:Graph, prefix='', sep='/', allow_repeated=False)->dict[str,Any]:
#     print(d_or_val)
#     if isinstance(d_or_val, dict):
#         result = {}
#         if prefix =="":
#             prefix_sep = ""
#         else:
#             prefix_sep = prefix+sep
#         for k, v in d_or_val.items():
#             new_prefix = prefix_sep+str(k)
#             flattened_child = flatten_dict(v,new_prefix,sep)
#             if not allow_repeated :
#                 intersection = intersect_lists(flattened_child.keys(),d_or_val.keys())
#                 if len(intersection)>0:
#                     raise DuplicateKeyError(intersection)
#             result.update(flattened_child)
#         return result
#     else :
#         return { prefix : d_or_val }

def flatten_dict_list(d_or_val:Graph[T],prefix="",full_name=True)->list[tuple[str,T]]:
    if isinstance(d_or_val, dict):
        result = []
        for k, v in d_or_val.items():
            if full_name:
                new_prefix = prefix + "/" + k
            else:
                new_prefix = k
            flattened_child = flatten_dict_list(v,prefix=new_prefix)
            result+=flattened_child
        return result
    else:
        return [(prefix, d_or_val)]
