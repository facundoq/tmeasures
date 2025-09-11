from asyncio import QueueFull
from queue import Empty, Queue
from time import sleep
from typing import Iterable, Sized


class FullQueue(Exception):
    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)


class IterableQueue(Sized, Iterable):
    """Queue supporting Iterator and Sized protocols.
    Queue has a finite size so it can be iterated upon with a for loop
    """

    def __init__(self, n: int, blocking_size=None,name=""):
        if blocking_size is None:
            blocking_size = n
        self.queue = Queue(maxsize=blocking_size)
        self.name=name
        self.n = n
        self.stop=False
        self.extra=0
        self.reset()

    def reset(self):
        self.removed= 0
        self.added=self.extra
        self.extra=0

    def __len__(self):
        return self.n

    def put(self, x,block=True,timeout=None):

        #     raise FullQueue(f"Queue {self.name} already has {self.n} items, no more can be added.")
        if self.full():
            self.extra+=1
        else:
            self.added+=1
        self.queue.put(x,block=block,timeout=timeout)

    def __iter__(self):
        return self

    def get(self,block=True,timeout=None):
        return self.queue.get(block=block,timeout=timeout)

    def fully_consumed(self):
        return self.removed == self.n

    def full(self):
        return self.added == self.n

    def __next__(self):
        # print(f"next {self.i}/{self.n} (maxsize {self.maxsize})")
        if self.fully_consumed() or self.stop:
            if not self.stop:
                self.reset()
            raise StopIteration()
        self.removed += 1

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
