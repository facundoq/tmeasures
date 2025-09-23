from __future__ import annotations

from typing import Any,TypeVar,TypeAlias
import typing


T = TypeVar('T')


Graph:TypeAlias = dict[str,'Graph[T]'] | T


def indices_of(l:list,value)->list[int]:
    return [i for i,l in enumerate(l) if value == l]

def get_all(l:list[T],indices:list[int])->list[T]:
    return [l[i] for i in indices]

def intersect_lists(a:list[T],b:list[T])->list[T]:
    return list(set(a) & set(b))

class DuplicateKeyError(ValueError):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

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
