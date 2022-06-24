from typing import Iterable


def producer():
    while True:
        a = yield
        return a


class Mean:
    def __init__(self,iterator):

        self.iterator =iterator
        next(self.iterator)

    def compute(self):
        m=0
        i=0
        for v in self.iterator:
            print(v)
            v+=m
            i+=1
        self.result = m

def generate_producers():
    n = 2
    values = list(range(1,n+1))
    means = [Mean() for i in values]

    for i in range(5):
        for i,m in enumerate(means):
            m.iterator.send(values[i])
        for i,v in enumerate(values):
            values[i]*=2
    for m in means:
        m.iterator.close()
        print(m.result)


if __name__ == '__main__':
    generate_producers()
