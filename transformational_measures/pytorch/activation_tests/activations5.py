import threading
import time
from threading import Thread
from transformational_measures.utils.iterable_queue import IterableQueue
import numpy as np
import torch
# class ActivationIterator:

#
#     def __next__(self):

def consumer(activations):
    print(f"Expecting {len(activations)} elements.. " )
    for row,row_activations in enumerate(activations):
        print(f"Row {row} with  {len(row_activations)} elements.. ")

        for col,activation in enumerate(row_activations):
            time.sleep(1)
            print(id(activation))

if __name__ == '__main__':
    n = 3
    m = 4
    n_queues = 2
    qs = [IterableQueue(n) for _ in range(n_queues)]
    threads = [threading.Thread(target=consumer, args=(q,)) for q in qs]
    for x in threads:
        x.start()
    val =torch.Tensor([1,2,3])
    print(id(val))
    for i in range(n):
        inner_qs = [IterableQueue(m) for _ in range(m)]
        for q,inner_q in zip(qs,inner_qs):
            q.put(inner_q)

        for j in range(m):
            for q2 in inner_qs:
                q2.put(val)
    for t in threads:
        t.join()