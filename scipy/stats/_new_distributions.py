from functools import cached_property
import numpy as np
from scipy import special
from scipy.stats._distribution_infrastructure import (
    ContinuousDistribution, _RealDomain, _RealParameter, _Parameterization,
    oo, _null)


class Normal(ContinuousDistribution):

    _x_support = _RealDomain(endpoints=(-oo, oo), inclusive=(True, True))
    _x_param = _RealParameter('x', domain=_x_support, typical=(-5, 5))
    _variable = _x_param

    def _logpdf(self, x, **kwargs):
        return -(np.log(2*np.pi)/2 + x**2/2)

    def _pdf(self, x, **kwargs):
        return 1/np.sqrt(2*np.pi) * np.exp(-x**2/2)

    def _logcdf(self, x, **kwargs):
        return special.log_ndtr(x)

    def _cdf(self, x, **kwargs):
        return special.ndtr(x)

    def _logccdf(self, x, **kwargs):
        return special.log_ndtr(-x)

    def _ccdf(self, x, **kwargs):
        return special.ndtr(-x)

    def _icdf(self, x, **kwargs):
        return special.ndtri(x)

    def _ilogcdf(self, x, **kwargs):
        return special.ndtri_exp(x)

    def _iccdf(self, x, **kwargs):
        return -special.ndtri(x)

    def _ilogccdf(self, x, **kwargs):
        return -special.ndtri_exp(x)

    def _entropy(self, **kwargs):
        return (1 + np.log(2*np.pi))/2

    def _logentropy(self, **kwargs):
        return np.log1p(np.log(2*np.pi)) - np.log(2)

    def _median(self, **kwargs):
        return 0

    def _mode(self, **kwargs):
        return 0

    def _moment_raw(self, order, **kwargs):
        raw_moments = {0: 1, 1: 0, 2: 1, 3: 0, 4: 3, 5: 0}
        return raw_moments.get(order, None)

    def _moment_central(self, order, **kwargs):
        return self._moment_raw(order, **kwargs)

    def _moment_standard(self, order, **kwargs):
        return self._moment_raw(order, **kwargs)

    def _sample(self, sample_shape, full_shape, rng, **kwargs):
        return rng.normal(size=full_shape)


def _log_diff(log_p, log_q):
    return special.logsumexp([log_p, log_q+np.pi*1j], axis=0)


class LogUniform(ContinuousDistribution):

    _a_domain = _RealDomain(endpoints=(0, oo))
    _b_domain = _RealDomain(endpoints=('a', oo))
    _log_a_domain = _RealDomain(endpoints=(-oo, oo))
    _log_b_domain = _RealDomain(endpoints=('log_a', oo))
    _x_support = _RealDomain(endpoints=('a', 'b'), inclusive=(True, True))

    _a_param = _RealParameter('a', domain=_a_domain, typical=(1e-3, 0.9))
    _b_param = _RealParameter('b', domain=_b_domain, typical=(1.1, 1e3))
    _log_a_param = _RealParameter('log_a', symbol=r'\log(a)',
                                  domain=_log_a_domain, typical=(-3, -0.1))
    _log_b_param = _RealParameter('log_b', symbol=r'\log(b)',
                                  domain=_log_b_domain, typical=(0.1, 3))
    _x_param = _RealParameter('x', domain=_x_support, typical=('a', 'b'))

    _b_domain.define_parameters(_a_param)
    _log_b_domain.define_parameters(_log_a_param)
    _x_support.define_parameters(_a_param, _b_param)

    _parameterizations = [_Parameterization(_log_a_param, _log_b_param),
                          _Parameterization(_a_param, _b_param)]
    _variable = _x_param

    @classmethod
    def _process_parameters(cls, a=None, b=None, log_a=None, log_b=None):
        a = np.exp(log_a) if a is None else a
        b = np.exp(log_b) if b is None else b
        log_a = np.log(a) if log_a is None else log_a
        log_b = np.log(b) if log_b is None else log_b
        return dict(a=a, b=b, log_a=log_a, log_b=log_b)

    # def _logpdf(self, x, *, log_a, log_b, **kwargs):
    #     return -np.log(x) - np.log(log_b - log_a)

    def _pdf(self, x, *, log_a, log_b, **kwargs):
        return ((log_b - log_a)*x)**-1

    # def _cdf(self, x, *, log_a, log_b, **kwargs):
    #     return (np.log(x) - log_a)/(log_b - log_a)

    def _moment_raw(self, order, log_a, log_b, **kwargs):
        if order == 0:
            return 1
        t1 = 1 / (log_b - log_a) / order
        t2 = np.real(np.exp(_log_diff(order * log_b, order * log_a)))
        return t1 * t2
