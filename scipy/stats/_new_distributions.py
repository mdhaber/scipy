import numpy as np
from scipy import special
from scipy.stats._distribution_infrastructure import (
    ContinuousDistribution, _RealDomain, _RealParameter, _Parameterization,
    oo, TransformedDistribution, _combine_docs)

__all__ = ['Normal', 'Uniform']

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
    r"""Normal distribution with prescribed mean and standard deviation.

    The probability density function of the normal distribution is:

    .. math::

        f(x) = \frac{1}{\sigma \sqrt{2 \pi}} \exp {
            \left( -\frac{1}{2}\left( \frac{x - \mu}{\sigma} \right)^2 \right)}

    """
    # `ShiftedScaledDistribution` allows this to be generated automatically from
    # an instance of `StandardNormal`, but the normal distribution is so frequently
    # used that it's worth a bit of code duplication to get better performance.
    _mu_domain = _RealDomain(endpoints=(-oo, oo))
    _sigma_domain = _RealDomain(endpoints=(0, oo))
    _x_support = _RealDomain(endpoints=(-oo, oo))

    _mu_param = _RealParameter('mu',  symbol=r'µ', domain=_mu_domain,
                               typical=(-1, 1))
    _sigma_param = _RealParameter('sigma', symbol=r'σ', domain=_sigma_domain,
                                  typical=(0.5, 1.5))
    _x_param = _RealParameter('x', domain=_x_support, typical=(-1, 1))

    _parameterizations = [_Parameterization(_mu_param, _sigma_param)]

    _variable = _x_param
    _normalization = 1/np.sqrt(2*np.pi)
    _log_normalization = np.log(2*np.pi)/2

    def __new__(cls, mu=None, sigma=None, **kwargs):
        if mu is None and sigma is None:
            return StandardNormal(**kwargs)
        return super().__new__(cls)

    def __init__(self, *, mu=0., sigma=1., **kwargs):
        super().__init__(mu=mu, sigma=sigma, **kwargs)

    def _logpdf_formula(self, x, *, mu, sigma, **kwargs):
        return StandardNormal._logpdf_formula(self, (x - mu)/sigma) - np.log(sigma)

    def _pdf_formula(self, x, *, mu, sigma, **kwargs):
        return StandardNormal._pdf_formula(self, (x - mu)/sigma) / sigma

    def _logcdf_formula(self, x, *, mu, sigma, **kwargs):
        return StandardNormal._logcdf_formula(self, (x - mu)/sigma)

    def _cdf_formula(self, x, *, mu, sigma, **kwargs):
        return StandardNormal._cdf_formula(self, (x - mu)/sigma)

    def _logccdf_formula(self, x, *, mu, sigma, **kwargs):
        return StandardNormal._logccdf_formula(self, (x - mu)/sigma)

    def _ccdf_formula(self, x, *, mu, sigma, **kwargs):
        return StandardNormal._ccdf_formula(self, (x - mu)/sigma)

    def _icdf_formula(self, x, *, mu, sigma, **kwargs):
        return StandardNormal._icdf_formula(self, x) * sigma + mu

    def _ilogcdf_formula(self, x, *, mu, sigma, **kwargs):
        return StandardNormal._ilogcdf_formula(self, x) * sigma + mu

    def _iccdf_formula(self, x, *, mu, sigma, **kwargs):
        return StandardNormal._iccdf_formula(self, x) * sigma + mu

    def _ilogccdf_formula(self, x, *, mu, sigma, **kwargs):
        return StandardNormal._ilogccdf_formula(self, x) * sigma + mu

    def _entropy_formula(self, *, mu, sigma, **kwargs):
        return StandardNormal._entropy_formula(self) + np.log(abs(sigma))

    def _logentropy_formula(self, *, mu, sigma, **kwargs):
        lH0 = StandardNormal._logentropy_formula(self)
        lls = np.log(np.log(abs(sigma))+0j)
        return special.logsumexp(np.broadcast_arrays(lH0, lls), axis=0)

    def _median_formula(self, *, mu, sigma, **kwargs):
        return mu

    def _mode_formula(self, *, mu, sigma, **kwargs):
        return mu

    def _moment_raw_formula(self, order, *, mu, sigma, **kwargs):
        if order == 0:
            return np.ones_like(mu)
        elif order == 1:
            return mu
        else:
            return None
    _moment_raw_formula.orders = [0, 1]

    def _moment_central_formula(self, order, *, mu, sigma, **kwargs):
        if order == 0:
            return np.ones_like(mu)
        elif order % 2:
            return np.zeros_like(mu)
        else:
            # exact is faster (and obviously more accurate) for reasonable orders
            return sigma**order * special.factorial2(int(order) - 1, exact=True)

    def _sample_formula(self, sample_shape, full_shape, rng, *, mu, sigma, **kwargs):
        return rng.normal(loc=mu, scale=sigma, size=full_shape)[()]


def _log_diff(log_p, log_q):
    return special.logsumexp([log_p, log_q+np.pi*1j], axis=0)


class StandardNormal(Normal):
    r"""Standard normal distribution.

    The probability density function of the standard normal distribution is:

    .. math::

        f(x) = \frac{1}{\sqrt{2 \pi}} \exp \left( -\frac{1}{2} x^2 \right)

    """
    _x_support = _RealDomain(endpoints=(-oo, oo))
    _x_param = _RealParameter('x', domain=_x_support, typical=(-5, 5))
    _variable = _x_param
    _parameterizations = []
    _normalization = 1/np.sqrt(2*np.pi)
    _log_normalization = np.log(2*np.pi)/2
    mu = np.float64(0.)
    sigma = np.float64(1.)

    def __new__(cls, *args, **kwargs):
        return ContinuousDistribution.__new__(cls)

    def __init__(self, **kwargs):
        ContinuousDistribution.__init__(self, **kwargs)

    def _logpdf_formula(self, x, **kwargs):
        return -(self._log_normalization + x**2/2)

    def _pdf_formula(self, x, **kwargs):
        return self._normalization * np.exp(-x**2/2)

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

    def _moment_standardized_formula(self, order, **kwargs):
        return self._moment_raw_formula(order, **kwargs)

    def _sample_formula(self, sample_shape, full_shape, rng, **kwargs):
        return rng.normal(size=full_shape)[()]


class LogUniform(ContinuousDistribution):
    r"""Log-uniform distribution.

    The probability density function of the log-uniform distribution is:

    ..math::

        f(x; a, b) = \frac{1}
                          {x (\log(b) - \log(a))}

    If :math:`\log(X)` is a random variable that follows a uniform distribution
    between :math:`\log(a)` and :math:`\log(b)`, then :math:`X` is log-uniformly
    distributed with shape parameters :math:`a` and :math:`b`.

    """

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

    def __init__(self, *, a=None, b=None, log_a=None, log_b=None, **kwargs):
        super().__init__(a=a, b=b, log_a=log_a, log_b=log_b, **kwargs)

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
            return self._one
        t1 = self._one / (log_b - log_a) / order
        t2 = np.real(np.exp(_log_diff(order * log_b, order * log_a)))
        return t1 * t2


class LogLaplace(ContinuousDistribution):
    """Log-Laplace distribution."""

    _mu_domain = _RealDomain(endpoints=(-oo, oo))
    _b_domain = _RealDomain(endpoints=(0, oo))
    _x_support = _RealDomain(endpoints=(0, np.inf), inclusive=(False, False))

    _mu_param = _RealParameter('mu', domain=_mu_domain, symbol='µ', typical=(-1e2, 1e2))
    _b_param = _RealParameter('b', domain=_b_domain, typical=(1, 10))
    _x_param = _RealParameter('x', domain=_x_support, typical=(1, 10))

    _parameterizations = [_Parameterization(_mu_param, _b_param)]
    _variable = _x_param

    def _pdf_formula(self, x, *, mu, b, **kwargs):
        return 1/(2*b*x) * np.exp(-np.abs(np.log(x) - mu)/b)

    def _moment_raw_formula(self, order, *, mu, b, **kwargs):
        with np.errstate(divide='ignore'):
            c2, n2 = b**-2, order**2
            return np.where(n2 < c2, c2 / (c2 - n2), np.inf)


class Uniform(ContinuousDistribution):
    r"""Uniform distribution.

    The probability density function of the uniform distribution is:

    ..math::

        f(x; a, b) = \frac{1}
                          {b - a}

    """

    _a_domain = _RealDomain(endpoints=(-oo, oo))
    _b_domain = _RealDomain(endpoints=('a', oo))
    _x_support = _RealDomain(endpoints=('a', 'b'), inclusive=(False, False))

    _a_param = _RealParameter('a', domain=_a_domain, typical=(1e-3, 0.9))
    _b_param = _RealParameter('b', domain=_b_domain, typical=(1.1, 1e3))
    _x_param = _RealParameter('x', domain=_x_support, typical=('a', 'b'))

    _b_domain.define_parameters(_a_param)
    _x_support.define_parameters(_a_param, _b_param)

    _parameterizations = [_Parameterization(_a_param, _b_param)]
    _variable = _x_param

    def __init__(self, *, a=None, b=None, **kwargs):
        super().__init__(a=a, b=b, **kwargs)

    def _process_parameters(self, a=None, b=None, ab=None, **kwargs):
        ab = b - a
        kwargs.update(dict(a=a, b=b, ab=ab))
        return kwargs

    def _pdf_formula(self, x, *, ab, **kwargs):
        return np.full(x.shape, 1/ab)

    def _icdf_formula(self, x, a, b, ab, **kwargs):
        return a + ab*x

    def _mode_formula(self, *, a, b, ab, **kwargs):
        return a + 0.5*ab

# class CircularDistribution(ShiftedScaledDistribution):
#     """Class that represents a circular statistical distribution."""
#     # Define 2-arg cdf functions
#     # Define 2-arg inverse CDF - one argument is left quantile
#     # Define mean, median, mode, variance, standard_deviation, entropy
#     # Raise error on use of moment functions?
#     # Should support be -inf, inf because that is the range of values that
#     #  produce nonzero pdf? Or should support be the left and right wrap
#     #  points? The trouble with left and right wrap points is that this
#     #  triggers `_set_invalid_nan` to zero the pdf. We'd need to adjust
#     #  `_set_invalid_nan` for circular distributions. (We probably need to
#     #  do that anyway.) The nice thing about using the left and right wrap
#     #  points is that some other methods would begin to do sensible things
#     #  by default. For example, I think `qmc_sample` would begin to work.
#     _a_domain = _RealDomain(endpoints=(-oo, oo), inclusive=(True, True))
#     _a_param = _RealParameter('a', domain=_a_domain)
#
#     _b_domain = _RealDomain(endpoints=('a', oo), inclusive=(True, True))
#     _b_param = _RealParameter('b', domain=_b_domain)
#
#     _parameterizations = [_Parameterization(_a_param, _b_param)]
#
#     def _process_parameters(self, a, b, **kwargs):
#         scale = b - a
#         parameters = self._dist._process_parameters(**kwargs)
#         parameters.update(dict(a=a, b=b, scale=scale))
#         return parameters
#
#     def _transform(self, x, a, b, scale, **kwargs):
#         x01 = (x - a)/scale  # shift/scale to 0-1
#         x01 %= 1  # wrap to 0-1
#         return 2*np.pi*x01 - np.pi  # shift/scale to -π, π
#
#     def _itransform(self, x, a, b, scale, **kwargs):
#         x01 = (x + np.pi)/(2*np.pi)  # shift/scale to 0-1
#         return scale*x01 + a  # shift/scale to a, b
#
#     def _support(self, a, b, scale, **kwargs):
#         return np.full_like(a, -np.inf), np.full_like(b, np.inf)
#
#     def _pdf_dispatch(self, x, *args, a, b, scale, **kwargs):
#         x = self._transform(x, a, b, scale)
#         pdf = self._dist._pdf_dispatch(x, *args, **kwargs)
#         return pdf / abs(scale) * 2*np.pi
#
#     def _sample_dispatch(self, sample_shape, full_shape, *,
#                          method, rng, **kwargs):
#         rvs = self._dist._sample_dispatch(
#             sample_shape, full_shape, method=method, rng=rng, **kwargs)
#         return self._itransform(rvs, **kwargs)
#

# class VonMises(CircularDistribution):
#     def __init__(self, *args, mu, kappa, a=-np.pi, b=np.pi, **kwargs):
#         super().__init__(_VonMises(mu=mu, kappa=kappa), *args,
#                          a=a, b=b, **kwargs)
#
#
# class _VonMises(ContinuousDistribution):
#
#     _mu_domain = _RealDomain(endpoints=(-np.pi, np.pi), inclusive=(True, True))
#     _kappa_domain = _RealDomain(endpoints=(0, oo), inclusive=(False, False))
#
#     _mu_param = _RealParameter('mu', symbol='µ', domain=_mu_domain,
#                                typical=(-1, 1))
#     _kappa_param = _RealParameter('kappa', symbol='κ', domain=_kappa_domain,
#                                   typical=(0.1, 10))
#     _x_param = _RealParameter('x', domain=_mu_domain, typical=(-1, 1))
#
#     _parameterizations = [_Parameterization(_mu_param, _kappa_param)]
#     _variable = _x_param
#
#     def _pdf_formula(self, x, mu, kappa, **kwargs):
#         return np.exp(kappa * np.cos(x - mu))/(2*np.pi*special.i0(kappa))
#
#     def _sample_formula(self, sample_shape, full_shape, rng, mu, kappa, **kwargs):
#         return rng.vonmises(mu=mu, kappa=kappa, size=full_shape)[()]


Normal.__doc__ = _combine_docs(Normal)
# LogUniform.__doc__ = _combine_docs(LogUniform)
