import numpy as np


class RunningMeanSingle:
    def __repr__(self):
        return f"RunningMeanSingle(n={self.n},mean={self.mean()})"

    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0

    def clear(self):
        self.n = 0

    def update(self, x):
        self.n += 1
        if self.n == 1:
            self.old_m = self.new_m =x
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.old_m = self.new_m

    def mean(self):
        return self.new_m if self.n else 0.0

class RunningMeanSimple:
    def __repr__(self):
        return f"RunningMeanSimple(n={self.n},mean={self.sum.shape})"

    def __init__(self):
        self.clear()

    def clear(self):
        self.n = 0
        self.sum = np.zeros(1)

    def update(self, x):
        self.n += 1
        self.sum+=x

    def update_all(self,x:np.ndarray):
        self.sum += x.sum(axis=0)
        self.n += x.shape[0]

    def mean(self):
        return self.sum if self.n > 0 else np.zeros(1)

class RunningMeanWelford:
    def __repr__(self):
        return f"RunningMean(n={self.n},mean={self.mean().shape})"

    def __init__(self):
        self.clear()

    def clear(self):
        self.n = 0
        self.mu = np.zeros(1,dtype=np.float32)

    def update(self, x):
        self.n += 1

        if self.n == 1:
            self.mu = x
        else:
            self.mu = self.mu + (x - self.mu) / self.n

    def update_all(self,x:np.ndarray):
        k=x.shape[0]
        if self.n == 0:
            self.mu = x.mean(axis=0)
            self.n = k
        else:
            self.n += k
            x_sum=x.sum(axis=0)
            self.mu = self.mu + (x_sum - self.mu * k) / self.n

    def mean(self):
        return self.mu if self.n > 0 else np.zeros(1)
import abc

class RunningMeanAndVariance(abc.ABC):
    @abc.abstractmethod
    def mean(self):
        pass

    @abc.abstractmethod
    def std(self):
        pass

    @abc.abstractmethod
    def var(self):
        pass

    @abc.abstractmethod
    def clear(self):
        pass

    @abc.abstractmethod
    def update(self,x):
        pass

    @abc.abstractmethod
    def update_all(self,x):
        pass
class RunningMeanAndVarianceNaive(RunningMeanAndVariance):
    def __repr__(self):
        return f"RunningMeanAndVarianceNaive(n={self.n},mean={self.mean().shape})"
    def __init__(self):
        self.e=RunningMeanWelford()
        self.e2=RunningMeanWelford()

    @property
    def n(self):
        return self.e.n

    def clear(self):
        self.e.clear()
        self.e2.clear()
    def mean(self):
        return self.e.mean()
    def std(self):
        return self.e2.mean() - self.e.mean()**2
    def update(self,x):
        self.e.update(x)
        self.e2.update(x*x)
    def var(self):
        s=self.std()
        return s*s
    def update_all(self,x):
        self.e.update_all(x)
        self.e2.update_all(x*x)

class RunningMeanAndVarianceWelford(RunningMeanAndVariance):

    def __repr__(self):

        return f"RunningMeanAndVarianceWellford(n={self.n},mean={self.mean().shape})"

    def __init__(self):
        self.n = 0
        self.m = np.array([0],dtype=np.float32)
        self.s = np.array([0],dtype=np.float32)

    def clear(self):
        self.n = 0

    def update(self, x):
        self.n += 1

        if self.n == 1:
            self.m = x
            self.s = np.zeros_like(self.m,dtype=np.float32)
        else:
            # diff = x - self.old_m
            # self.new_m = self.old_m + (x - self.old_m) / self.n
            # self.s = self.s + diff * (x - self.new_m)
            # self.old_m = self.new_m

            diff = x - self.m
            self.m += diff / self.n
            self.s +=  diff * (x - self.m)

    # def update_all(self,x:np.ndarray):
    # see https://stackoverflow.com/questions/56402955/whats-the-formula-for-welfords-algorithm-for-variance-std-with-batch-updates
    #     k=x.shape[0]
    #     self.n += 1
    #
    #     if self.n == 1:
    #         self.m = x.mean(axis=0)
    #         self.s = x.std(axis=0)
    #     else:
    #         # diff = x - self.old_m
    #         # self.new_m = self.old_m + (x - self.old_m) / self.n
    #         # self.s = self.s + diff * (x - self.new_m)
    #         # self.old_m = self.new_m
    #
    #         diff = x - self.m
    #         self.m += diff / self.n
    #         self.s += diff * (x - self.m)

    def update_all(self,x:np.ndarray):
        for i in range(x.shape[0]):
            self.update(x[i,:])

    def mean(self):
        return self.m if self.n else np.zeros_like(self.m)

    def var(self):
        return self.s / (self.n - 1) if self.n > 1 else np.zeros_like(self.s)

    def std(self):
        return np.sqrt(self.var())
