from functools import cached_property
import numpy as np
from scipy import special
from scipy.stats._distribution_infrastructure import (
    ContinuousDistribution, _RealDomain, _RealParameter, _Parameterization,
    oo, _null)


class Normal(ContinuousDistribution):

    _x_support = _RealDomain(endpoints=(-oo, oo), inclusive=(False, False))
    _x_param = _RealParameter('x', domain=_x_support, typical=(-5, 5))
    _parameterizations = []
    _variable = _x_param

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

    def _moment_raw(self, order, **kwargs):
        raw_moments = {0: 1, 1: 0, 2: 1, 3: 0, 4: 3, 5: 0}
        return raw_moments.get(order, None)

    def _moment_central(self, order, **kwargs):
        return self._moment_raw(order, **kwargs)

    def _moment_standard(self, order, **kwargs):
        return self._moment_raw(order, **kwargs)


from scipy import special
def _log_diff(log_p, log_q):
    return special.logsumexp([log_p, log_q+np.pi*1j], axis=0)

class LogUniform(ContinuousDistribution):

    _a_domain = _RealDomain(endpoints=(0, oo))
    _b_domain = _RealDomain(endpoints=('a', oo))
    _log_a_domain = _RealDomain(endpoints=(-oo, oo))
    _log_b_domain = _RealDomain(endpoints=('log_a', oo))
    _x_support = _RealDomain(endpoints=('a', 'b'), inclusive=(True, True))

    _a_param = _RealParameter('a', domain=_a_domain, typical=(1e-3, 1))
    _b_param = _RealParameter('b', domain=_b_domain, typical=(1, 1e3))
    _log_a_param = _RealParameter('log_a', symbol=r'\log(a)',
                                  domain=_log_a_domain, typical=(-3, 0))
    _log_b_param = _RealParameter('log_b', symbol=r'\log(b)',
                                  domain=_log_b_domain, typical=(0, 3))
    _x_param = _RealParameter('x', domain=_x_support, typical=('a', 'b'))

    _b_domain.define_parameters(_a_param)
    _log_b_domain.define_parameters(_log_a_param)
    _x_support.define_parameters(_a_param, _b_param)

    _parameterizations = [_Parameterization(_log_a_param, _log_b_param),
                          _Parameterization(_a_param, _b_param)]
    _variable = _x_param

    def __init__(self, *, a=_null, b=_null, log_a=_null, log_b=_null, **kwargs):
        super().__init__(a=a, b=b, log_a=log_a, log_b=log_b, **kwargs)

    @cached_property
    def a(self):
        return (self._parameters['a'] if 'a' in self._parameters
                else np.exp(self._parameters['log_a']))

    @cached_property
    def b(self):
        return (self._parameters['b'] if 'b' in self._parameters
                else np.exp(self._parameters['log_b']))

    @cached_property
    def log_a(self):
        return (self._parameters['log_a'] if 'log_a' in self._parameters
                else np.log(self._parameters['a']))

    @cached_property
    def log_b(self):
        return (self._parameters['log_b'] if 'log_b' in self._parameters
                else np.log(self._parameters['b']))

    # def _logpdf(self, x, *, log_a, log_b, **kwargs):
    #     return -np.log(x) - np.log(log_b - log_a)

    # def _moment_raw(self, order, **kwargs):
    #     if order == 0:
    #         return np.nan
    #     else:
    #         return None

    def _pdf(self, x, *, log_a, log_b, **kwargs):
        return ((log_b - log_a)*x)**-1

    def _moment_raw(self, order, log_a, log_b, **kwargs):
        if order == 0:
            return 1
        t1 = 1 / (log_b - log_a) / order
        t2 = np.real(np.exp(_log_diff(order * log_b, order * log_a)))
        return t1 * t2
    #
    # def _cdf(self, x, *, log_a, log_b, **kwargs):
    #     return (np.log(x) - log_a)/(log_b - log_a)