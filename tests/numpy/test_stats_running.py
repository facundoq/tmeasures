from tmeasures.np.stats_running import RunningMeanAndVarianceWelford, RunningMeanWelford,RunningMean,RunningVariance,RunningMeanVarianceSets,RunningMeanSets
import pytest

import numpy as np
import itertools
eps = 1e-5
atol = 0.1
dimensions=[10]
n_samples = [100,1000]
scales = [1e-5, 1, 100]
centers = [-5, 0, 10]
batch_size = 10
for n in n_samples:
    assert n % batch_size == 0

values = [np.random.randn(n,dim)*s+c for (n,dim,s,c) in itertools.product(n_samples,dimensions,scales,centers)]

values_batched =[np.random.randn(n//batch_size,batch_size,dim)*s+c for (n,dim,s,c) in itertools.product(n_samples,dimensions,scales,centers)]

values = [v.astype(np.double) for v in values]
values_batched = [v.astype(np.double) for v in values_batched]

def compare_variance_batched(m:RunningVariance,x:np.ndarray):

    for i in range(x.shape[0]):
        m.update_batch(x[i,:, :])
    expected = x.var(axis=(0,1),ddof=1)
    actual = m.var()
    assert np.sum(actual < 0) == 0, "Variance can't be negative"
    # assert expected == pytest.approx(actual,rel=eps,abs=atol)
    np.testing.assert_allclose(expected, actual, rtol=eps,atol=atol)

def compare_variance(m:RunningVariance,x:np.ndarray):
    
    for i in range(x.shape[0]):
        m.update(x[i, :])

    expected = x.var(axis=0,ddof=1)
    actual = m.var()
    assert np.sum(actual < 0) == 0, "Variance can't be negative"
    np.testing.assert_allclose(expected, actual, rtol=eps)


def compare_means_batched(m:RunningMean,x:np.ndarray):
    for i in range(x.shape[0]):
        m.update_batch(x[i,:, :])
    expected = x.mean(axis=(0,1))
    actual = m.mean()
    np.testing.assert_allclose(expected, actual, rtol=eps)

def compare_means(m:RunningMean,x:np.ndarray):
    for i in range(x.shape[0]):
        m.update(x[i, :])
    expected = x.mean(axis=0)
    actual = m.mean()
    np.testing.assert_allclose(expected, actual, rtol=eps,atol=atol)

@pytest.mark.parametrize("x",values)
def test_mean_welford(x:np.ndarray):
    compare_means(RunningMeanWelford(),x)

@pytest.mark.parametrize("x",values_batched)
def test_mean_welford(x:np.ndarray):
    compare_means_batched(RunningMeanWelford(),x)

@pytest.mark.parametrize("x",values)
def test_mean_sets(x:np.ndarray):
    compare_means(RunningMeanSets(),x)

@pytest.mark.parametrize("x",values_batched)
def test_mean_sets_batched(x:np.ndarray):
    compare_means_batched(RunningMeanSets(),x)

@pytest.mark.parametrize("x",values)
def test_mean_variance_welford(x:np.ndarray):
    compare_means(RunningMeanAndVarianceWelford(),x)

@pytest.mark.parametrize("x",values_batched)
def test_mean_variance_welford(x:np.ndarray):
    compare_means_batched(RunningMeanAndVarianceWelford(),x)

@pytest.mark.parametrize("x",values)
def test_variance_sets(x:np.ndarray):
    compare_variance(RunningMeanVarianceSets(),x)

@pytest.mark.parametrize("x",values_batched)
def test_variance_sets_batched(x:np.ndarray):
    compare_variance_batched(RunningMeanVarianceSets(),x)

@pytest.mark.parametrize("x",values)
def test_variance_welford(x:np.ndarray):
    compare_variance(RunningMeanAndVarianceWelford(),x)

@pytest.mark.parametrize("x",values_batched)
def test_variance_welford(x:np.ndarray):
    compare_variance_batched(RunningMeanAndVarianceWelford(),x)
