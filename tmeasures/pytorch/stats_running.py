import torch
import abc


class RunningMeasure(abc.ABC):
    @abc.abstractmethod
    def clear(self):
        pass

    @abc.abstractmethod
    def update(self, x: torch.Tensor):
        pass

    def update_all(self, x: torch.Tensor):
        for i in range(x.shape[0]):
            self.update(x[i, :])

    @abc.abstractmethod
    def update_batch(self, x: torch.Tensor):
        pass


class RunningMean(RunningMeasure):

    @abc.abstractmethod
    def mean(self):
        pass


class RunningVariance(RunningMeasure):
    @abc.abstractmethod
    def var(self):
        pass

    def std(self):
        return torch.sqrt(self.var())


class RunningMeanWelford(RunningMean):
    def __repr__(self):
        return f"RunningMean(n={self.n},mean={self.mean().shape})"

    def __init__(self):
        self.clear()

    def clear(self):
        self.n = 0
        self.mu = 0

    def update(self, x: torch.Tensor):
        self.n += 1
        self.mu += (x - self.mu) / self.n

    def update_all(self, x: torch.Tensor):
        self.update_batch(x)

    def update_batch(self, x: torch.Tensor):
        k = x.shape[0]
        self.n += k
        diff = x - self.mu
        self.mu += diff.sum(dim=0) / self.n

    def mean(self):
        return self.mu if self.n > 0 else torch.zeros(1)


class RunningMeanAndVarianceWelford(RunningMean, RunningVariance):

    def __repr__(self):
        return f"RunningMeanAndVarianceWellford(n={self.n},mean={self.mean().shape})"

    def __init__(self):
        self.clear()

    def clear(self):
        self.n = 0
        self.m = 0
        self.s = 0

    def update(self, x):
        # x = x.double()
        self.n += 1
        diff = x - self.m
        self.m = self.m + (diff / self.n)
        self.s = self.s + diff * (x - self.m)

    def update_batch(self, x: torch.Tensor):
        # x = x.double()
        # see https://hackmd.io/_-rPOUx5RpajaZZkyU9wgw
        k = x.shape[0]
        self.n += k
        diff = x - self.m
        self.m += diff.sum(dim=0) / self.n
        diff2 = (diff * (x - self.m)).sum(dim=0)
        self.s += diff2

    def mean(self):
        # assert(self.n>0)
        return self.m if self.n else torch.zeros_like(self.m)

    def var(self):
        # assert (self.n > 1)
        return self.s / (self.n - 1) if self.n > 1 else torch.zeros_like(self.s)



class RunningMeanSets(RunningMean):
    def __repr__(self):
        return f"RunningMeanSets(n={self.n},mean={self.mean().shape})"

    def __init__(self):
        self.clear()

    def clear(self):
        self.n = 0
        self.mu = 0

    def mean(self):
        return self.mu

    def update_batch(self, x: torch.Tensor):
        m = x.shape[0]
        mu_x = x.mean(dim=0)
        n = self.n
        self.n = m + n
        if n == 0:
            self.mu = mu_x
        else:
            c1, c2 = n / self.n, m / self.n
            self.mu = c1 * self.mu + c2 * mu_x

    def update(self, x: torch.Tensor):
        self.n += 1
        if self.n == 1:
            self.mu = x
        else:
            self.mu = self.mu + (x - self.mu) / self.n


# noinspection PyAttributeOutsideInit
class RunningMeanVarianceSets(RunningMean, RunningVariance):
    def __repr__(self):
        return f"RunningMeanVarianceSets(n={self.n},mean={self.mean().shape})"

    def __init__(self):
        self.clear()

    def clear(self):
        self.n = 0
        self.mu = 0
        self.v = 0

    def update_batch(self, x: torch.Tensor):

        m = x.shape[0]
        mu_x = x.mean(dim=0)
        if m>1:
            v_x = x.var(dim=0, unbiased = False)
        else:
            v_x=0
        n = self.n
        self.n = m + n
        if n == 0:
            self.mu = mu_x
            self.v = v_x
        else:
            c1, c2 = n / self.n, m / self.n
            c3 = c1*c2
            # print(c1,c2,c3)
            mu = self.mu
            self.mu = c1 * self.mu + c2 * mu_x
            self.v = c1 * self.v + c2 * v_x + c3 * (mu - mu_x) ** 2
            # print(self.v)

    def var(self):
        return self.v*self.n/(self.n-1)

    def mean(self):
        return self.mu

    def update(self, x: torch.Tensor):
        self.update_batch(torch.unsqueeze(x,dim=0))

