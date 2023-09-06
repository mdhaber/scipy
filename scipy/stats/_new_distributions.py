from functools import cached_property
import numpy as np
from scipy import special
from scipy.stats._distribution_infrastructure import (
    ContinuousDistribution, _RealDomain, _RealParameter, _Parameterization,
    oo, _null, ShiftedScaledDistribution, TransformedDistribution)


def factorial(n):
    return special.gamma(n + 1)


class OrderStatisticDistribution(TransformedDistribution):

    # These should really be _IntegerDomain/_IntegerParameter
    _r_domain = _RealDomain(endpoints=(1, 'n'), inclusive=(True, True))
    _r_param = _RealParameter('r', domain=_r_domain, typical=(1, 2))

    _n_domain = _RealDomain(endpoints=(1, np.inf), inclusive=(True, True))
    _n_param = _RealParameter('n', domain=_n_domain, typical=(1, 4))

    _r_domain.define_parameters(_n_param)

    _parameterizations = [_Parameterization(_r_param, _n_param)]

    def _overrides(self, method_name):
        return method_name == '_pdf_formula'

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

    def _process_parameters(self, a=None, b=None, log_a=None, log_b=None, **kwargs):
        a = np.exp(log_a) if a is None else a
        b = np.exp(log_b) if b is None else b
        log_a = np.log(a) if log_a is None else log_a
        log_b = np.log(b) if log_b is None else log_b
        kwargs.update(dict(a=a, b=b, log_a=log_a, log_b=log_b))
        return kwargs

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


class CircularDistribution(ShiftedScaledDistribution):
    # Define 2-arg cdf functions
    # Define 2-arg inverse CDF - one argument is left quantile
    # Define mean, median, mode, var, std, entropy
    # Raise error on use of moment functions?
    # Should support be -inf, inf because that is the range of values that
    #  produce nonzero pdf? Or should support be the left and right wrap
    #  points? The trouble with left and right wrap points is that this
    #  triggers `_set_invalid_nan` to zero the pdf. We'd need to adjust
    #  `_set_invalid_nan` for circular distributions. (We probably need to
    #  do that anyway.) The nice thing about using the left and right wrap
    #  points is that some other methods would begin to do sensible things
    #  by default. For example, I think `qmc_sample` would begin to work.
    _a_domain = _RealDomain(endpoints=(-oo, oo), inclusive=(True, True))
    _a_param = _RealParameter('a', domain=_a_domain)

    _b_domain = _RealDomain(endpoints=('a', oo), inclusive=(True, True))
    _b_param = _RealParameter('b', domain=_b_domain)

    _parameterizations = [_Parameterization(_a_param, _b_param)]

    def _process_parameters(self, a, b, **kwargs):
        scale = b - a
        parameters = self._dist._process_parameters(**kwargs)
        parameters.update(dict(a=a, b=b, scale=scale))
        return parameters

    def _transform(self, x, a, b, scale, **kwargs):
        x01 = (x - a)/scale  # shift/scale to 0-1
        x01 %= 1  # wrap to 0-1
        return 2*np.pi*x01 - np.pi  # shift/scale to -π, π

    def _itransform(self, x, a, b, scale, **kwargs):
        x01 = (x + np.pi)/(2*np.pi)  # shift/scale to 0-1
        return scale*x01 + a  # shift/scale to a, b

    def _support(self, a, b, scale, **kwargs):
        return np.full_like(a, -np.inf), np.full_like(b, np.inf)

    def _pdf_dispatch(self, x, *args, a, b, scale, **kwargs):
        x = self._transform(x, a, b, scale)
        pdf = self._dist._pdf_dispatch(x, *args, **kwargs)
        return pdf / abs(scale) * 2*np.pi

    def _sample_dispatch(self, sample_shape, full_shape, *,
                         method, rng, **kwargs):
        rvs = self._dist._sample_dispatch(
            sample_shape, full_shape, method=method, rng=rng, **kwargs)
        return self._itransform(rvs, **kwargs)


class VonMises(CircularDistribution):
    def __init__(self, *args, mu, kappa, a=-np.pi, b=np.pi, **kwargs):
        super().__init__(_VonMises(mu=mu, kappa=kappa), *args,
                         a=a, b=b, **kwargs)


class _VonMises(ContinuousDistribution):

    _mu_domain = _RealDomain(endpoints=(-np.pi, np.pi), inclusive=(True, True))
    _kappa_domain = _RealDomain(endpoints=(0, oo), inclusive=(False, False))

    _mu_param = _RealParameter('mu', symbol='µ', domain=_mu_domain,
                               typical=(-1, 1))
    _kappa_param = _RealParameter('kappa', symbol='κ', domain=_kappa_domain,
                                  typical=(0.1, 10))
    _x_param = _RealParameter('x', domain=_mu_domain, typical=(-1, 1))

    _parameterizations = [_Parameterization(_mu_param, _kappa_param)]
    _variable = _x_param

    def _pdf_formula(self, x, mu, kappa, **kwargs):
        return np.exp(kappa * np.cos(x - mu))/(2*np.pi*special.i0(kappa))

    def _sample_formula(self, sample_shape, full_shape, rng, mu, kappa, **kwargs):
        return rng.vonmises(mu=mu, kappa=kappa, size=full_shape)[()]
