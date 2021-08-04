import numpy as np
import abc


class RunningMeasure(abc.ABC):
    @abc.abstractmethod
    def clear(self):
        pass

    @abc.abstractmethod
    def update(self, x: np.ndarray):
        pass

    def update_all(self, x: np.ndarray):
        for i in range(x.shape[0]):
            self.update(x[i, :])

    @abc.abstractmethod
    def update_batch(self, x: np.ndarray):
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
        return np.sqrt(self.var())


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

    def update_batch(self, x: np.ndarray):
        m = x.shape[0]
        mu_x = x.mean(axis=0)
        n = self.n
        self.n = m + n
        if n == 0:
            self.mu = mu_x
        else:
            c1, c2 = n / self.n, m / self.n
            self.mu = c1 * self.mu + c2 * mu_x

    def update(self, x: np.ndarray):
        self.n += 1
        if self.n == 1:
            self.mu = x
        else:
            self.mu = self.mu + (x - self.mu) / self.n


# noinspection PyAttributeOutsideInit
class RunningMeanVarianceSets(RunningMean, RunningVariance):
    def __repr__(self):
        return f"RunningMeanVarianceSets(n={self.n})"

    def __init__(self):
        self.clear()

    def clear(self):
        self.n = 0
        self.mu = 0
        self.v = 0

    def update_batch(self, x: np.ndarray):
        m = x.shape[0]
        mu_x = x.mean(axis=0)
        if m==1:
            v_x=0
        else:
            # print(x)
            # import matplotlib.pyplot as plt
            # f,ax=plt.subplots(x.shape[0])
            # for i in range(x.shape[0]):
            #     im = x[i,10,:,:]
            #     # im = im.transpose((1,2,))
            #
            #     ax[i].imshow(im)
            # plt.savefig(f"batch{self.n}.png")
            # print(x[:,4:6,16,14])
            v_x = x.var(axis=0,ddof=0)
            # print(v_x[:,14:16,14:16])

        n = self.n
        self.n = m + n
        # print("pre")
        if n == 0:
            self.mu = mu_x
            # print(m,v_x)
            self.v = v_x
        else:
            c1, c2= n / self.n, m / self.n
            c3 = c1 * c2
            mu = self.mu
            self.mu = c1 * self.mu + c2 * mu_x
            self.v = c1 * self.v + c2 * v_x + c3 * ((mu - mu_x) ** 2)

    def var(self):
        return self.v*self.n/(self.n-1)

    def mean(self):
        return self.mu

    def update(self, x: np.ndarray):
        self.update_batch(x[np.newaxis,])


class RunningMeanWelford(RunningMean):
    def __repr__(self):
        return f"RunningMeanWelford(n={self.n},mean={self.mean().shape})"

    def __init__(self):
        self.clear()

    def clear(self):
        self.n = 0
        self.mu = np.zeros(1, dtype=np.float32)

    def update(self, x):
        self.n += 1

        if self.n == 1:
            self.mu = x
        else:
            self.mu = self.mu + (x - self.mu) / self.n

    def update_batch(self, x: np.ndarray):
        k = x.shape[0]
        if self.n == 0:
            self.mu = x.mean(axis=0)
            self.n = k
        else:
            self.n += k
            x_sum = x.sum(axis=0)
            self.mu = self.mu + (x_sum - self.mu * k) / self.n

    def mean(self):
        return self.mu if self.n > 0 else np.zeros(1)


class RunningMeanAndVarianceWelford(RunningMean, RunningVariance):

    def __repr__(self):
        return f"RunningMeanAndVarianceWelford(n={self.n},mean={self.mean().shape})"

    def __init__(self):
        self.clear()

    def clear(self):
        self.n = 0
        self.m = 0
        self.s = 0

    def update(self, x: np.ndarray):
        self.n += 1
        diff = x - self.m
        self.m = self.m + (diff / self.n)
        self.s = self.s + diff * (x - self.m)

    def update_batch(self, x: np.ndarray):
        # see https://stackoverflow.com/questions/56402955/whats-the-formula-for-welfords-algorithm-for-variance-std-with-batch-updates
        k = x.shape[0]

        self.n += k
        diff = x - self.m
        self.m = self.m + diff.sum(axis=0) / self.n
        # print(diff * (x - self.m))
        # print(self.m[:10,0,0])
        diff2 = (diff * (x - self.m)).sum(axis=0)
        self.s = self.s + diff2
        # print("s=",self.s.min(),self.s.max())

    def mean(self):
        return self.m if self.n > 0 else np.zeros_like(self.m)

    def var(self):
        return self.s / (self.n - 1) if self.n > 1 else np.zeros_like(self.s)

