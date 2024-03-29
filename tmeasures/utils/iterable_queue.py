from queue import Queue, Empty
from time import sleep
from typing import Sized, Iterable


class IterableQueue(Sized, Iterable):
    """Queue supporting Iterator and Sized protocols.
    Queue has a max size so it can be iterated upon with a for loop
    """

    def __init__(self, n: int, maxsize=None,name=""):
        if maxsize is None:
            maxsize = n
        self.queue = Queue(maxsize=maxsize)
        self.name=name
        self.n = n
        self.i = 0
        self.stop=False

    def __len__(self):
        return self.n

    def put(self, x,block=True,timeout=None):
        self.queue.put(x,block=block,timeout=timeout)

    def __iter__(self):
        self.i = 0
        return self
    def get(self,block=True,timeout=None):
        return self.queue.get(block=block,timeout=timeout)
    
    def __next__(self):
        # print(f"next {self.i}/{self.n} (maxsize {self.maxsize})")
        self.i += 1
        if self.i == self.n + 1 or self.stop:
            raise StopIteration()
        # print(f"ITQUEUE {self.name}: Getting item...")
        # item = None
        # while item is None:
        #     try:
        #         item = self.queue.get_nowait()
        #     except Empty:
        #         # print(f"{self.name} sleeping")
        #         sleep(0)
        # print(f"ITQUEUE {self.name}: obtained item: {item}...")
        item = self.queue.get()
        return item
