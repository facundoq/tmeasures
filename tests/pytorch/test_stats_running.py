from tmeasures.pytorch.stats_running import RunningMeanAndVarianceWelford, RunningMeanWelford,RunningMean,RunningVariance,RunningMeanSets,RunningMeanVarianceSets
import itertools
import torch
import pytest

eps = 1e-5
abs = 1e-2
dimensions=[10]
n_samples = [100,1000]
scales = [ 1e-5, 1, 10]
centers = [-5, 0, 10]
batch_size = 10
for n in n_samples:
    assert n % batch_size == 0

values = [torch.randn(n,dim)*s+c for (n,dim,s,c) in itertools.product(n_samples,dimensions,scales,centers)]
values_batched =[torch.randn(n//batch_size,batch_size,dim)*s+c for (n,dim,s,c) in itertools.product(n_samples,dimensions,scales,centers)]

# use doubles
values = [v.double() for v in values]
values_batched = [v.double() for v in values_batched]

def compare_values(expected:torch.Tensor,actual:torch.Tensor):
    print("Expected:")
    print(expected)
    print("Actual:")
    print(actual)

    return torch.all(torch.isclose(expected,actual,rtol=eps))


def compare_variance_batched(m:RunningVariance,x:torch.Tensor):
    for i in range(x.shape[0]):
        m.update_batch(x[i,:, :])
    expected = x.var(dim=(0,1))
    actual = m.var()
    assert torch.sum(actual < 0) == 0, f"Variance can't be negative {actual}"
    torch.testing.assert_allclose(actual,expected,rtol=eps,atol=abs)


def compare_variance(m:RunningVariance,x:torch.Tensor):
    for i in range(x.shape[0]):
        m.update(x[i, :])
    expected = x.var(dim=0)
    actual = m.var()
    assert torch.sum(actual < 0) == 0, f"Variance can't be negative {actual}"
    torch.testing.assert_allclose(actual, expected, rtol=eps, atol=abs)


def compare_means(m:RunningMean,x:torch.Tensor):
    for i in range(x.shape[0]):
        m.update(x[i, :])
    expected = x.mean(dim=0)
    actual = m.mean()
    torch.testing.assert_allclose(actual, expected, rtol=eps, atol=abs)


def compare_means_batched(m:RunningMean,x:torch.Tensor):
    for i in range(x.shape[0]):
        m.update_batch(x[i,:, :])
    expected = x.mean(dim=(0,1))
    actual = m.mean()
    torch.testing.assert_allclose(actual, expected, rtol=eps, atol=abs)


@pytest.mark.parametrize("x", values)
def test_mean_welford(x: torch.Tensor):
    compare_means(RunningMeanWelford(), x)


@pytest.mark.parametrize("x", values_batched)
def test_mean_welford(x: torch.Tensor):
    compare_means_batched(RunningMeanWelford(), x)


@pytest.mark.parametrize("x", values)
def test_mean_sets(x: torch.Tensor):
    compare_means(RunningMeanSets(), x)


@pytest.mark.parametrize("x", values_batched)
def test_mean_sets_batched(x: torch.Tensor):
    compare_means_batched(RunningMeanSets(), x)


@pytest.mark.parametrize("x", values)
def test_mean_variance_welford(x: torch.Tensor):
    compare_means(RunningMeanAndVarianceWelford(), x)


@pytest.mark.parametrize("x", values_batched)
def test_mean_variance_welford(x: torch.Tensor):
    compare_means_batched(RunningMeanAndVarianceWelford(), x)


@pytest.mark.parametrize("x", values)
def test_variance_sets(x: torch.Tensor):
    compare_variance(RunningMeanVarianceSets(), x)


@pytest.mark.parametrize("x", values_batched)
def test_variance_sets_batched(x: torch.Tensor):
    print(f"Shape of input:{x.shape}, (μ={x.mean(dim=(0,1))},σ={x.var(dim=(0,1))}\n")
    compare_variance_batched(RunningMeanVarianceSets(), x)

@pytest.mark.parametrize("x",values)
def test_variance_welford(x:torch.Tensor):
    compare_variance(RunningMeanAndVarianceWelford(),x)

@pytest.mark.parametrize("x",values_batched)
def test_variance_welford(x:torch.Tensor):
    compare_variance_batched(RunningMeanAndVarianceWelford(),x)
