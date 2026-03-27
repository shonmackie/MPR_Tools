import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from numpy import random, isclose, array, inf, empty, mean, std, sqrt, linspace, concatenate

from MPR_Tools import PerformanceAnalyzer
from MPR_Tools.core.matter_interactions import ProbabilityDistribution


def test_gaussian_fwhm():
    rng = random.default_rng(seed=0)
    x = rng.normal(loc=100, scale=10, size=10000)
    width, center = PerformanceAnalyzer.fwfm(x, fractional_max=1/2)
    assert isclose(width, 23.55, atol=0.3, rtol=0.1)  # the standard deviation is about 30, so the random error is about 0.3
    assert isclose(center, 100, atol=0.3, rtol=0.1)  # and there can be significant relative errors due to broadening from the KDE


def test_long_tailed_gaussian_fwhm():
    rng = random.default_rng(seed=0)
    x = concatenate([
        rng.normal(loc=100, scale=10, size=10000),
        rng.normal(loc=100, scale=100, size=10000),  # this distribution forms long tails that shouldn't affect the width much
    ])
    width, center = PerformanceAnalyzer.fwfm(x, fractional_max=1/2)
    assert isclose(width, 25.25, atol=0.3, rtol=0.1)  # I found this 25.25 numerically; idk if there's an analytic solution
    assert isclose(center, 100, atol=0.3, rtol=0.1)


def test_uniform_fwhm():
    rng = random.default_rng(seed=0)
    x = rng.uniform(low=10, high=30, size=10000)
    width, center = PerformanceAnalyzer.fwfm(x, fractional_max=1/2)
    assert isclose(width, 20, atol=0.3, rtol=0.1)
    assert isclose(center, 20, atol=0.3, rtol=0.1)


def test_gamma_fwhm():
    rng = random.default_rng(seed=0)
    x = rng.gamma(shape=1/2, scale=10, size=10000)
    width, center = PerformanceAnalyzer.fwfm(x, fractional_max=1/2)
    assert 0 < width < 10  # the width of this distribution isn't really defined, but any reasonable estimate will be < 10
    assert isclose(center, width/2, atol=0, rtol=1e-3)


def test_probability_distribution():
    # the probability distribution is defined by the linear interpolation between these three points
    x_table = array([10, 20, 30])
    p_table = array([20, 10, 20])
    distribution = ProbabilityDistribution(x_table, p_table)
    
    assert distribution.integral(20, inf) == 1/2
    
    rng = random.default_rng(0)
    samples = empty(10000)
    for i in range(samples.size):
        samples[i] = distribution.draw(rng, lower=20, upper=inf)
    assert all((samples >= 20) & (samples <= 30))
    assert isclose(mean(samples), 20 + 50/9, atol=3*0.003)  # the standard deviation is about 0.3, so random error is about 0.003
    assert isclose(std(samples), 10*sqrt(13/162), atol=3*0.003)
    
    plt.figure()
    plt.hist(samples, density=True, bins=50)
    x = linspace(20, 30)
    plt.plot(x, 2/30*(1 + (x - 20)/10), '--')
    plt.title("Parabolic probability distribution")
    plt.tight_layout()
    plt.savefig("tests/output/figures/test_probability_distribution.png")
    plt.close()
