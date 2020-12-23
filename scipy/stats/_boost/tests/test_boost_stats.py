'''Boost stats tests.'''

import pytest
import numpy as np
import scipy.stats
from scipy.stats.boost import (
    beta as boost_beta,
    nbinom as boost_nbinom,
    binom as boost_binom,
)


def test_issue_10317():
    alpha, n, p = 0.9, 10, 1
    assert boost_nbinom.interval(alpha=alpha, n=n, p=p) == (0, 0)

def test_issue_11134():
    alpha, n, p = 0.95, 10, 0
    assert boost_binom.interval(alpha=alpha, n=n, p=p) == (0, 0)

def test_issue_7406():
    np.random.seed(0)
    assert np.all(boost_binom.ppf(np.random.rand(10), 0, 0.5) == 0)

def test_binom_ppf_endpoints():
    assert boost_binom.ppf(0, 0, 0.5) == -1
    assert boost_binom.ppf(1, 0, 0.5) == 0

def test_issue_5122():
    p = 0
    n = np.random.randint(100, size=10)

    x = 0
    ppf = boost_binom.ppf(x, n, p)
    assert np.all(ppf == -1)

    x = np.linspace(0.01, 0.99, 10)
    ppf = boost_binom.ppf(x, n, p)
    assert np.all(ppf == 0)

    x = 1
    ppf = boost_binom.ppf(x, n, p)
    assert np.all(ppf == n)

def test_issue_1603():
    assert np.all(boost_binom(1000, np.logspace(-3, -100)).ppf(0.01) == 0)

def test_issue_5503():
    p = 0.5
    x = np.logspace(3, 14, 12)
    assert np.allclose(boost_binom.cdf(x, 2*x, p), 0.5, atol=1e-2)

@pytest.mark.parametrize('x, n, p, cdf_desired', [
    (300, 1000, 3/10, 0.51559351981411995636),
    (3000, 10000, 3/10, 0.50493298381929698016),
    (30000, 100000, 3/10, 0.50156000591726422864),
    (300000, 1000000, 3/10, 0.50049331906666960038),
    (3000000, 10000000, 3/10, 0.50015600124585261196),
    (30000000, 100000000, 3/10, 0.50004933192735230102),
    (30010000, 100000000, 3/10, 0.98545384016570790717),
    (29990000, 100000000, 3/10, 0.01455017177985268670),
    (29950000, 100000000, 3/10, 5.02250963487432024943e-28),
])
def test_issue_5503pt2(x, n, p, cdf_desired):
    assert np.allclose(boost_binom.cdf(x, n, p), cdf_desired)

def test_issue_5503pt3():
    # From Wolfram Alpha: CDF[BinomialDistribution[1e12, 1e-12], 2]
    assert np.allclose(boost_binom.cdf(2, 10**12, 10**-12), 0.91969860292869777384)
