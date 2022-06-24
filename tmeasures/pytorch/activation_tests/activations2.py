from typing import Iterable

class ConsumerIterator(Iterable):
    def __init__(self,generator):
        self.generator=generator

    def __iter__(self):
        return self

    def __next__(self):
        a = yield from self.generator
        return a

def producer():
    a = yield
    return a

class Mean:
    def compute(self,iterator):
        m=0
        i=0
        for v in iterator:
           v+=m
           i+=1
        self.result = m

def generate_producers():
    n = 3
    values = list(range(1,n+1))
    producers = [producer() for i in values]
    # initialize producers
    for c in producers:
        next(c)
    iterators = [ConsumerIterator(p) for p in producers]
    means = [Mean() for i in iterators]
    for m,i in zip(means,iterators):
        print(m.compute(i))
    for i in range(5):
        for i,c in enumerate(producers):
            c.send(values[i])
        for i,v in enumerate(values):
            values[i]*=2
    for m in means:
        print(m.result)


if __name__ == '__main__':
    generate_producers()
