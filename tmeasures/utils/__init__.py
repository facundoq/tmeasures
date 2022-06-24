
def indices_of(list:[],value)->[int]:
    indices =[]
    for i,l in enumerate(list):
        if value == l:
            indices.append(i)
    return indices

def get_all(list:[],indices:[int])->[]:
    return [list[i] for i in indices]




from queue import Queue
from typing import Sized,Iterable

class IterableQueue(Queue,Sized,Iterable):
    """Queue supporting Iterator and Sized protocols.
    Queue has a max size so it can be iterated upon with a for loop
    """
    def __init__(self, n:int,maxsize=None):
        if maxsize is None:
            maxsize=n
        super().__init__(maxsize)
        self.n=n
        self.i=0

    def __len__(self):
        return self.n


    def __iter__(self):
        return self

    def __next__(self):
        self.i+=1
        if self.i == self.n:
            raise StopIteration()
        return self.get()