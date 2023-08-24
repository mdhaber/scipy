from functools import cached_property
import numpy as np
from scipy import special
from scipy.stats._distribution_infrastructure import (
    ContinuousDistribution, _RealDomain, _RealParameter, _Parameterization,
    oo, _null, ShiftedScaledDistribution)


def factorial(n):
    return special.gamma(n + 1)


class OrderStatisticDistribution(ContinuousDistribution):

    # These should really be _IntegerDomain/_IntegerParameter
    _r_domain = _RealDomain(endpoints=(1, 'n'), inclusive=(True, True))
    _r_param = _RealParameter('r', domain=_r_domain, typical=(1, 2))

    _n_domain = _RealDomain(endpoints=(1, np.inf), inclusive=(True, True))
    _n_param = _RealParameter('n', domain=_n_domain, typical=(1, 4))

    _r_domain.define_parameters(_n_param)

    _parameterizations = [_Parameterization(_r_param, _n_param)]

    def __init__(self, *args, dist, **kwargs):
        # This needs some careful thought, but I think it can work.
        self._dist = dist
        self._variable = dist._variable
        super().__init__(*args, **kwargs)
        self._parameters.update(dist._parameters)
        self._shape = np.broadcast_shapes(self._shape, dist._shape)
        self._invalid = np.broadcast_to(self._invalid, self._shape)

    def _pdf_formula(self, x, r, n, **kwargs):
        factor = factorial(n) / (factorial(r-1) * factorial(n-r))
        fX = self._dist._pdf_dispatch(x, **kwargs)
        FX = self._dist._cdf_dispatch(x, **kwargs)
        cFX = self._dist._ccdf_dispatch(x, **kwargs)
        return factor * fX * FX**(r-1) * cFX**(n-r)


class Normal(ContinuousDistribution):

    _x_support = _RealDomain(endpoints=(-oo, oo), inclusive=(True, True))
    _x_param = _RealParameter('x', domain=_x_support, typical=(-5, 5))
    _variable = _x_param
    normalization = 1/np.sqrt(2*np.pi)
    log_normalization = np.log(2*np.pi)/2

    def _logpdf_formula(self, x, **kwargs):
        return -(self.log_normalization + x**2/2)

    def _pdf_formula(self, x, **kwargs):
        return self.normalization * np.exp(-x**2/2)

    def _logcdf_formula(self, x, **kwargs):
        return special.log_ndtr(x)

    def _cdf_formula(self, x, **kwargs):
        return special.ndtr(x)

    def _logccdf_formula(self, x, **kwargs):
        return special.log_ndtr(-x)

    def _ccdf_formula(self, x, **kwargs):
        return special.ndtr(-x)

    def _icdf_formula(self, x, **kwargs):
        return special.ndtri(x)

    def _ilogcdf_formula(self, x, **kwargs):
        return special.ndtri_exp(x)

    def _iccdf_formula(self, x, **kwargs):
        return -special.ndtri(x)

    def _ilogccdf_formula(self, x, **kwargs):
        return -special.ndtri_exp(x)

    def _entropy_formula(self, **kwargs):
        return (1 + np.log(2*np.pi))/2

    def _logentropy_formula(self, **kwargs):
        return np.log1p(np.log(2*np.pi)) - np.log(2)

    def _median_formula(self, **kwargs):
        return 0

    def _mode_formula(self, **kwargs):
        return 0

    def _moment_raw_formula(self, order, **kwargs):
        raw_moments = {0: 1, 1: 0, 2: 1, 3: 0, 4: 3, 5: 0}
        return raw_moments.get(order, None)

    def _moment_central_formula(self, order, **kwargs):
        return self._moment_raw_formula(order, **kwargs)

    def _moment_standard_formula(self, order, **kwargs):
        return self._moment_raw_formula(order, **kwargs)

    def _sample_formula(self, sample_shape, full_shape, rng, **kwargs):
        return rng.normal(size=full_shape)[()]


class ShiftedScaledNormal(ShiftedScaledDistribution):
    _loc_domain = _RealDomain(endpoints=(-oo, oo), inclusive=(True, True))
    _loc_param = _RealParameter('loc', symbol='µ',
                                domain=_loc_domain, typical=(1, 2))

    _scale_domain = _RealDomain(endpoints=(-oo, oo), inclusive=(True, True))
    _scale_param = _RealParameter('scale', symbol='σ',
                                  domain=_scale_domain, typical=(0.1, 10))

    _parameterizations = [_Parameterization(_loc_param,
                                            _scale_param),
                          _Parameterization(_loc_param),
                          _Parameterization(_scale_param)]
    def __init__(self, *args, **kwargs):
        super().__init__(Normal(), *args, **kwargs)


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

    def _process_parameters(self, a=None, b=None, log_a=None, log_b=None):
        a = np.exp(log_a) if a is None else a
        b = np.exp(log_b) if b is None else b
        log_a = np.log(a) if log_a is None else log_a
        log_b = np.log(b) if log_b is None else log_b
        return dict(a=a, b=b, log_a=log_a, log_b=log_b)

    # def _logpdf_formula(self, x, *, log_a, log_b, **kwargs):
    #     return -np.log(x) - np.log(log_b - log_a)

    def _pdf_formula(self, x, *, log_a, log_b, **kwargs):
        return ((log_b - log_a)*x)**-1

    # def _cdf_formula(self, x, *, log_a, log_b, **kwargs):
    #     return (np.log(x) - log_a)/(log_b - log_a)

    def _moment_raw_formula(self, order, log_a, log_b, **kwargs):
        if order == 0:
            return 1
        t1 = 1 / (log_b - log_a) / order
        t2 = np.real(np.exp(_log_diff(order * log_b, order * log_a)))
        return t1 * t2
