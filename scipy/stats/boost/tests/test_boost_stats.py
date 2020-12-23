'''Boost stats tests.'''

import pytest
import numpy as np
import scipy.stats
from scipy.stats.boost import (
    beta as boost_beta,
    nbinom as boost_nbinom,
    binom as boost_binom,
)

def test_issue_12635():
    # Confirm that Boost's beta distribution resolves gh-12635. Check against R:
    # options(digits=16)
    # p = 0.9999999999997369
    # a = 75.0
    # b = 66334470.0
    # print(qbeta(p, a, b))
    p, a, b = 0.9999999999997369, 75.0, 66334470.0
    assert np.allclose(boost_beta.ppf(p, a, b), 2.343620802982393e-06)

def test_issue_12794():
    # Confirm that Boost's beta distribution resolves gh-12794. Check against R.
    # options(digits=16)
    # p = 1e-11
    # count_list = c(10,100,1000)
    # print(qbeta(1-p, count_list + 1, 100000 - count_list))
    inv_R = np.array([0.0004944464889611935,
                      0.0018360586912635726,
                      0.0122663919942518351])
    count_list = np.array([10, 100, 1000])
    p = 1e-11
    inv = boost_beta.isf(p, count_list + 1, 100000 - count_list)
    assert np.allclose(inv, inv_R)
    res = boost_beta.sf(inv, count_list + 1, 100000 - count_list)
    assert np.allclose(res, p)

def test_issue_12796():
    #Confirm that Boost's beta distribution succeeds in the case of gh-12796
    alpha_2 = 5e-6
    count_ = np.arange(1, 20)
    nobs = 100000
    q, a, b = 1 - alpha_2, count_ + 1, nobs - count_
    inv = boost_beta.ppf(q, a, b)
    res = boost_beta.cdf(inv, a, b)
    assert np.allclose(res, 1 - alpha_2)

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
