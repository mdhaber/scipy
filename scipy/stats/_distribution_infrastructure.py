from functools import cached_property
from scipy._lib._util import _lazywhere
from scipy import special
from scipy.integrate._tanhsinh import _tanhsinh
from scipy.optimize._zeros_py import _chandrupatla, _bracket_root
import numpy as np
_null = object()
oo = np.inf

# TODO:
#  documentation
#  figure out method call graph
#  tests
#  add methods
#  be more careful about passing **kwargs vs **self._all_shapes. For private
#   functions, it should always be the latter.
#  figure out caching - methods that don't accept shape parameters can be
#   cached, but then we couldn't change tolerance
#  pass shape parameters to methods that don't accept additional args?
#  pass atol, rtol to methods
#  profile/optimize
#  add array API support

# Originally, I planned to filter out invalid shape parameters for the
# author of the distribution; they would always work with "compressed",
# 1D arrays containing only valid shape parameters. There are two problems
# with this:
# - This essentially requires copying all arrays, even if there is only a
#   single invalid parameter combination. This is expensive. Then, to output
#   the original size data to the user, we need to "decompress" the arrays
#   and fill in the NaNs, so more copying. Unless we branch the code when
#   there are no invalid data, these copies happen even in the normal case,
#   where there are no invalid parameter combinations. We should not incur
#   all this overhead in the normal case.
# - For methods that accept arguments other than the shape parameters, the
#   user will pass in arrays that are broadcastable with the original arrays,
#   not the compressed arrays. This means that this same sort of invalid
#   value detection needs to be repeated every time one of these methods is
#   called.
#   The much simpler solution is to keep the data uncompressed but to replace
#   the invalid shape parameters and arguments with NaNs (and only if some are
#   invalid). With this approach, the copying happens only if/when it is
#   needed. Most functions involved in stats distribution calculations don't
#   mind NaNs; they just return NaN. The behavior "If x_i is NaN, the result
#   is NaN" is explicit in the array API. So this should be fine.
#   I'm also going to leave the data in the original shape. The reason for this
#   is that the user can process shape parameters as needed and make them
#   @cached_properties. If we leave all the original shapes alone, the input
#   to functions like `pdf` that accept additional arguments will be
#   broadcastable with these @cached_properties. In most cases, this is
#   completely transparent to the author.

def _set_invalid_nan(f):
    # need to come back to this to think more about use of < vs <=
    # update this
    use_support = {'logpdf', 'logcdf', 'logccdf', 'pdf', 'cdf', 'ccdf'}
    use_01 = {'icdf', 'iccdf'}
    replace_lows = {'logpdf': -oo, 'logcdf': -oo, 'logccdf': 0,
                   'pdf': 0, 'cdf': 0, 'ccdf': 1,
                   'icdf': np.nan, 'iccdf': np.nan,
                   'ilogcdf': np.nan, 'ilogccdf': np.nan}
    replace_highs = {'logpdf': -oo, 'logcdf': 0, 'logccdf': -oo,
                    'pdf': 0, 'cdf': 1, 'ccdf': 0,
                    'icdf': np.nan, 'iccdf': np.nan,
                    'ilogcdf': np.nan, 'ilogccdf': np.nan}

    def filtered(self, x, *args, skip_iv=False, **kwargs):
        if self.skip_iv or skip_iv:
            return f(self, x, *args, **kwargs)

        method_name = f.__name__
        if method_name in use_support:
            low, high = self.support
        elif method_name in use_01:
            low, high = 0, 1
        else:
            low, high = -np.inf, 0
        replace_low = replace_lows[method_name]
        replace_high = replace_highs[method_name]
        try:
            x, invalid, low, high = np.broadcast_arrays(x, self._invalid, low, high)
        except ValueError as e:
            message = (f"The argument provided to "
                       f"`{self.__class__.__name__}.{method_name}` cannot "
                       "be be broadcast to the same shape as the distribution "
                       "parameters.")
            raise ValueError(message) from e

        # check implications of <, <=
        x, valid = self._variable.check_dtype(x)
        mask_low = x < low if method_name == "pdf" else x <= low
        mask_high = x > high if method_name == "pdf" else x >= high
        x_invalid = (mask_low | mask_high | ~valid)
        if np.any(x_invalid):
            x = np.copy(x)
            x[x_invalid] = np.nan
        # TODO: ensure dtype is at least float and that output shape is correct
        out = np.asarray(f(self, x, *args, **kwargs))
        out[mask_low] = replace_low
        out[mask_high] = replace_high
        return out[()]

    return filtered


def kwargs2args(f, args=[], kwargs={}):
    # this is a temporary workaround until the scalar algorithms `_tanhsinh`,
    # `_chandrupatla`, etc., accept `kwargs` or can operate with compressing
    # arguments to the callable
    names = list(kwargs.keys())
    n_args = len(args)

    def wrapped(x, *args):
        return f(x, *args[:n_args], **dict(zip(names, args[n_args:])))

    args = list(args) + list(kwargs.values())

    return wrapped, args


def _solve_bounded(f, p, *, min, max, kwargs):
    # should modify _bracket_root and _chandrupatla so we don't need all this
    p, min, max = np.broadcast_arrays(p, min, max)
    def f2(x, p, **kwargs):
        return f(x, **kwargs) - p
    f3, args = kwargs2args(f2, args=[p], kwargs=kwargs)
    # If we know the median or mean, should use it

    # Any operations between 0d array and a scalar produces a scalar, so...
    shape = min.shape
    min, max = np.atleast_1d(min, max)

    a = -np.ones_like(min)
    b = np.ones_like(max)
    d = max - min

    i = np.isfinite(min) & np.isfinite(max)
    a[i] = min[i] + 0.25 * d[i]
    b[i] = max[i] - 0.25 * d[i]

    i = np.isfinite(min) & ~np.isfinite(max)
    a[i] = min[i] + 1
    b[i] = min[i] + 2

    i = np.isfinite(max) & ~np.isfinite(min)
    a[i] = max[i] - 2
    b[i] = max[i] - 1

    min = min.reshape(shape)
    max = max.reshape(shape)
    a = a.reshape(shape)
    b = b.reshape(shape)

    res = _bracket_root(f3, a=a, b=b, min=min, max=max, args=args)
    return _chandrupatla(f3, a=res.xl, b=res.xr, args=args)


def _log1mexp(x):
    def f1(x):
        return np.log1p(-np.exp(x))

    def f2(x):
        return np.real(np.log(-special.expm1(x + 0j)))

    return _lazywhere(x < -1, (x,), f=f1, f2=f2)


def _logexpxmexpy(x, y):
    x, y = np.broadcast_arrays(x, y)
    return special.logsumexp([x, y+np.pi*1j], axis=0)


class ContinuousDistribution:

    def __init__(self, *, tol=_null, skip_iv=False, **shapes):
        self.tol = tol
        self.skip_iv = skip_iv

        self._not_implemented = (
            f"`{self.__class__.__name__}` does not provide an accurate "
            "implementation of the required method. Leave `tol` unspecified "
            "to use the default implementation."
        )

        # identify parameterization
        all_shape_names = list(shapes)
        shapes = {key:val for key, val in shapes.items() if val is not _null}
        shape_names, shape_vals = zip(*shapes.items())
        for parameterization in self._parameterizations:
            if parameterization.validate(set(shape_names)):
                break
        else:
            message = (f"The provided shapes `{set(shape_names)}` "
                       "do not match a supported parameterization of the "
                       f"`{self.__class__.__name__}` distribution family.")
            raise ValueError(message)
        self._parameterization = parameterization

        if skip_iv:
            self._shapes = shapes
            self._all_shape_names = all_shape_names
            return

        # broadcast shape arguments
        try:
            shape_vals = np.broadcast_arrays(*shape_vals)
        except ValueError as e:
            message = (f"The shapes {set(shape_names)} provided to the "
                       f"`{self.__class__.__name__}` distribution family "
                       "cannot be broadcast to the same shape.")
            raise ValueError(message) from e

        # Replace invalid shapes with `np.nan`
        shapes = dict(zip(shape_names, shape_vals))
        self._invalid = ~parameterization.validate_shapes(shapes)
        # If necessary, make the arrays contiguous and replace invalid with NaN
        if np.any(self._invalid):
            for shape_name in shapes:
                shapes[shape_name] = np.copy(shapes[shape_name])
                shapes[shape_name][self._invalid] = np.nan

        self._shapes = shapes
        self._all_shape_names = all_shape_names

    def _get_shapes(self, shape_names=None):
        shape_names = shape_names or self._all_shape_names
        return {name: getattr(self, name) for name in shape_names}

    @cached_property
    def _all_shapes(self):
        # It would be better if we could pass to private methods (e.g. _pdf)
        # only the shapes that it wants. We could do this with inspection,
        # but it is probably faster to remember the names of the shapes needed
        # by each method.
        return self._get_shapes()

    @property
    def tol(self):
        return self._tol

    @tol.setter
    def tol(self, tol):
        if not (tol is _null or np.isscalar(tol)):
            message = (f"Parameter `tol` of {self.__class__.__name__} must be "
                       "a scalar, if specified.")
            raise ValueError(message)
        self._tol = tol

    @cached_property
    def support(self):
        return self._support(**self._all_shapes)

    def _support(self, **kwargs):
        a, b = self._variable.domain.endpoints
        a = kwargs.get(a, a)[()]
        b = kwargs.get(b, b)[()]
        return a, b

    def _overrides(self, method_name):
        method = getattr(self.__class__, method_name, None)
        super_method = getattr(self.__class__.__base__, method_name, None)
        return method is not super_method

    @_set_invalid_nan
    def logpdf(self, x):
        return self._logpdf(x, **self._all_shapes)

    @_set_invalid_nan
    def logcdf(self, x):
        return self._logcdf(x, **self._all_shapes)

    @_set_invalid_nan
    def logccdf(self, x):
        return self._logccdf(x, **self._all_shapes)

    def logentropy(self):
        return self._logentropy(**self._all_shapes)

    @_set_invalid_nan
    def ilogcdf(self, x):
        return self._ilogcdf(x, **self._all_shapes)

    @_set_invalid_nan
    def ilogccdf(self, x):
        return self._ilogccdf(x, **self._all_shapes)

    def logmoment(self, order, logcenter=None, standardized=False):
        # input validation
        logcenter = self.logmean if logcenter is None else logcenter
        raw = self._logmoment(order, logcenter, **self._all_shapes)
        res = raw - self.logvar*order/2 if standardized else raw
        return res

    @cached_property
    def logmean(self):
        return np.real(self._logmoment(1, -np.inf, **self._all_shapes))

    @cached_property
    def logvar(self):
        return self._logvar(logmean=self.logmean, **self._all_shapes)

    def _logvar(self, logmean, **kwargs):
        return np.real(self._logmoment(2, logmean, **kwargs))

    @cached_property
    def logstd(self):
        return self.logvar/2

    @cached_property
    def logskewness(self):
        return self._logskewness(logmean=self.logmean, logvar=self.logvar,
                                 **self._all_shapes)

    def _logskewness(self, logmean, logvar, **kwargs):
        return (np.real(self._logmoment(3, logmean, **kwargs)) - 1.5 * logvar)

    @cached_property
    def logkurtosis(self):
        return self._logkurtosis(logmean=self.logmean, logvar=self.logvar,
                                 **self._all_shapes)

    def _logkurtosis(self, logmean, logvar, **kwargs):
        return (np.real(self._logmoment(4, logmean, **kwargs)) - 2 * logvar)

    @_set_invalid_nan
    def pdf(self, x):
        return self._pdf(x, **self._all_shapes)

    @_set_invalid_nan
    def cdf(self, x):
        return self._cdf(x, **self._all_shapes)

    @_set_invalid_nan
    def ccdf(self, x):
        return self._ccdf(x, **self._all_shapes)

    def entropy(self):
        return self._entropy(**self._all_shapes)

    @_set_invalid_nan
    def icdf(self, x):
        return self._icdf(x, **self._all_shapes)

    @_set_invalid_nan
    def iccdf(self, x):
        return self._iccdf(x, **self._all_shapes)

    def sample(self, shape=(), rng=None):
        shape = (shape,) if not np.iterable(shape) else tuple(shape)
        rng = np.random.default_rng() if rng is None else rng
        return self._sample(shape, rng, **self._all_shapes)

    def _sample(self, shape, rng, **kwargs):
        full_shape = shape + self._invalid.shape
        uniform = rng.uniform(size=full_shape)
        return self._icdf(uniform, **kwargs)

    def moment(self, order, center=None, standardized=False):
        # Come back to this. Still needs a lot of work.
        # input validation / standardization
        # clean this up
        # ensure NaN pattern and array/scalar output
        # make sure we're using cache or override when possible and just
        # compute from scratch otherwise. I'm thinking there should be a
        # `_standard_moment` function that the developer can override, and we'd
        # manually cache results for all orders. `var`, `skewness`, etc., would
        # call this function. Too much logic is in `moment` itself.
        if order == 0:
            # replace with `np.ones` of correct shape/dtype and NaN pattern
            return np.ones_like(self.mean)[()]
        elif order == 1:
            central = np.zeros_like(self.mean)
        elif order == 2:
            central = self.var
        elif order == 3:
            central = self.skewness*self.var**1.5
        elif order == 4:
            central = self.kurtosis*self.var**2.

        if (order <= 3 or (order == 4 and self._overrides('_skewness'))):
            if center is not None:
                centrals = [self.moment(order=i) for i in range(order)] + [central]
                nonstandard = self._moment_transform_center(order, centrals,
                                                            self.mean, center)
            else:
                nonstandard = central
        else:
            center = self.mean if center is None else center
            nonstandard = self._moment(order, center, **self._all_shapes)

        moment = nonstandard / self.var ** (order / 2) if standardized else nonstandard

        moment = np.asarray(moment)
        moment[self._invalid] = np.nan
        return moment[()]

    def _moment_transform_center(self, order, moment_as, a, b):
        # a and b should be broadcasted before getting here
        # this is wrong - it's not just moment_a of order `order`; all lower
        # moments are needed, too
        n = order
        i = np.arange(n+1).reshape([-1]+[1]*a.ndim)  # orthogonal to other axes
        n_choose_i = special.binom(n, i)
        moment_b = np.sum(n_choose_i*moment_as*(a-b)**(n-i), axis=0)
        return moment_b

    @cached_property
    def mean(self):
        return self._mean(**self._all_shapes)

    def _mean(self, **kwargs):
        return self._moment(1, 0, **kwargs)

    @cached_property
    def std(self):
        return self.var**0.5

    @cached_property
    def var(self):
        return self._var(mean=self.mean, **self._all_shapes)

    def _var(self, mean, **kwargs):
        return self._moment(2, mean, **kwargs)

    @cached_property
    def skewness(self):
        return self._skewness(mean=self.mean, var=self.var, **self._all_shapes)

    def _skewness(self, mean, var, **kwargs):
        return self._moment(3, mean, **kwargs) / var ** 1.5

    @cached_property
    def kurtosis(self):
        # not Fisher kurtosis
        return self._kurtosis(mean=self.mean, var=self.var, **self._all_shapes)

    def _kurtosis(self, mean, var, **kwargs):
        return self._moment(4, mean, **kwargs)/var**2

    def _logpdf(self, x, **kwargs):
        if self.tol is _null:
            return np.log(self._pdf(x, **kwargs))
        else:
            raise NotImplementedError(self._not_implemented)

    def _pdf(self, x, **kwargs):
        if self._overrides('_logpdf'):
            return np.exp(self._logpdf(x, **kwargs))
        else:
            raise NotImplementedError(self._not_implemented)

    def _logcdf(self, x, **kwargs):
        if self.tol is _null and self._overrides('_cdf'):
            return np.log(self._cdf(x, **kwargs))
        elif self._overrides('_logccdf'):
            return self._logcdf_log1mexpccdf(x, **kwargs)
        else:
            return self._logcdf_integrate_logpdf(x, **kwargs)

    def _cdf(self, x, **kwargs):
        if self._overrides('_logcdf'):
            return np.exp(self._logcdf(x, **kwargs))
        elif self._tol is _null and self._overrides('_ccdf'):
            return self._cdf_1mccdf(x, **kwargs)
        else:
            return self._cdf_integrate_pdf(x, **kwargs)

    def _logccdf(self, x, **kwargs):
        if self.tol is _null and self._overrides('_cdf'):
            return np.log(self._ccdf(x, **kwargs))
        elif self._overrides('_logcdf'):
            return self._logccdf_log1mexpcdf(x, **kwargs)
        else:
            return self._logccdf_integrate_logpdf(x, **kwargs)

    def _ccdf(self, x, **kwargs):
        if self._overrides('_logccdf'):
            return np.exp(self._logccdf(x, **kwargs))
        elif self._tol is _null and self._overrides('_cdf'):
            return self._ccdf_1mcdf(x, **kwargs)
        else:
            return self._ccdf_integrate_pdf(x, **kwargs)

    def _logentropy(self, **kwargs):
        if self.tol is _null and self._overrides('_entropy'):
            return np.log(self._entropy(**kwargs))
        else:
            return self._logentropy_integrate_logpdf(**kwargs)

    def _logmoment(self, order, logcenter, **kwargs):
        return self._logmoment_integrate_logpdf(order, logcenter, **kwargs)

    def _entropy(self, **kwargs):
        if self._overrides('_logentropy'):
            return np.exp(self._logentropy(**kwargs))
        else:
            return self._entropy_integrate_pdf(**kwargs)

    def _icdf(self, x, **kwargs):
        if self.tol is _null and self._overrides('_iccdf'):
            return self._icdf_iccdf1m(x, **kwargs)
        else:
            return self._icdf_solve_cdf(x, **kwargs)

    def _iccdf(self, x, **kwargs):
        if self.tol is _null and self._overrides('_icdf'):
            return self._iccdf_icdf1m(x, **kwargs)
        else:
            return self._iccdf_solve_ccdf(x, **kwargs)

    def _ilogcdf(self, x, **kwargs):
        if self._overrides('_ilogccdf'):
            return self._ilogcdf_ilogccdf1m(x, **kwargs)
        else:
            return self._ilogcdf_solve_logcdf(x, **kwargs)

    def _ilogccdf(self, x, **kwargs):
        if self._overrides('_ilogcdf'):
            return self._ilogccdf_ilogcdf1m(x, **kwargs)
        else:
            return self._ilogccdf_solve_logccdf(x, **kwargs)

    def _moment(self, order, center, **kwargs):
        return self._moment_integrate_pdf(order, center, **kwargs)

    # Distribution functions via exp of log distribution functions
    def _entropy_exp_logentropy(self, **kwargs):
        return np.exp(self._logentropy(**kwargs))

    def _pdf_exp_logpdf(self, x, **kwargs):
        return np.exp(self._logpdf(x, **kwargs))

    def _cdf_exp_logcdf(self, x, **kwargs):
        return np.exp(self._logcdf(x, **kwargs))

    def _ccdf_exp_logccdf(self, x, **kwargs):
        return np.exp(self._logccdf(x, **kwargs))

    # Log distribution functions as log of distribution functions

    def _logentropy_log_entropy(self, **kwargs):
        return np.log(self._entropy(**kwargs))

    def _logpdf_log_pdf(self, x, **kwargs):
        return np.log(self._pdf(x, **kwargs))

    def _logcdf_log_cdf(self, x, **kwargs):
        return np.log(self._cdf(x, **kwargs))

    def _logccdf_log_logcdf(self, x, **kwargs):
        return np.log(self._ccdf(x, **kwargs))

    # Distribution functions via numerical integration

    def _logcdf_integrate_logpdf(self, x, **kwargs):
        a, b = self._support(**kwargs)
        f, args = kwargs2args(self._logpdf, kwargs=kwargs)
        res = _tanhsinh(f, a, x, args=args, log=True)
        return res.integral

    def _cdf_integrate_pdf(self, x, **kwargs):
        a, b = self._support(**kwargs)
        f, args = kwargs2args(self._pdf, kwargs=kwargs)
        res = _tanhsinh(f, a, x, args=args)
        return res.integral

    def _logccdf_integrate_logpdf(self, x, **kwargs):
        a, b = self._support(**kwargs)
        f, args = kwargs2args(self._logpdf, kwargs=kwargs)
        res = _tanhsinh(f, x, b, args=args, log=True)
        return res.integral

    def _ccdf_integrate_pdf(self, x, **kwargs):
        a, b = self._support(**kwargs)
        f, args = kwargs2args(self._pdf, kwargs=kwargs)
        res = _tanhsinh(f, x, b, args=args)
        return res.integral

    def _logentropy_integrate_logpdf(self, **kwargs):
        a, b = self.support
        f, args = kwargs2args(self._logpdf, kwargs=kwargs)
        def integrand(x, *args):
            logpdf = f(x, *args)
            return logpdf + np.log(0j+logpdf)
        res = _tanhsinh(integrand, a, b, args=args, log=True)
        return res.integral + np.pi*1j

    def _entropy_integrate_pdf(self, **kwargs):
        a, b = self.support
        f, args = kwargs2args(self._pdf, kwargs=kwargs)
        def integrand(x, *args):
            pdf = f(x, *args)
            return np.log(pdf)*pdf
        res = _tanhsinh(integrand, a, b, args=args)
        return -res.integral

    def _logmoment_integrate_logpdf(self, order, logcenter, **kwargs):
        a, b = self.support
        def logintegrand(x, order, logcenter, **kwargs):
            logpdf = self._logpdf(x, **kwargs)
            return logpdf + order*_logexpxmexpy(np.log(x+0j), logcenter)
        f, args = kwargs2args(logintegrand, args=(order, logcenter), kwargs=kwargs)
        res = _tanhsinh(f, a, b, args=args, log=True)
        return res.integral

    def _moment_integrate_pdf(self, order, center, **kwargs):
        a, b = self.support
        def integrand(x, order, center, **kwargs):
            pdf = self._pdf(x, **kwargs)
            return pdf*(x-center)**order
        f, args = kwargs2args(integrand, args=(order, center), kwargs=kwargs)
        res = _tanhsinh(f, a, b, args=args)
        return res.integral

    # Distribution functions as complement of other distribution functions

    def _cdf_1mccdf(self, x, **kwargs):
        return 1 - self._ccdf(x, **kwargs)

    def _ccdf_1mcdf(self, x, **kwargs):
        return 1 - self._cdf(x, **kwargs)

    def _logcdf_log1mexpccdf(self, x, **kwargs):
        return _log1mexp(self._logccdf(x, **kwargs))

    def _logccdf_log1mexpcdf(self, x, **kwargs):
        return _log1mexp(self._logcdf(x, **kwargs))

    def _icdf_iccdf1m(self, x, **kwargs):
        return self._iccdf(1-x, **kwargs)

    def _iccdf_icdf1m(self, x, **kwargs):
        return self._icdf(1-x, **kwargs)

    def _ilogcdf_ilogccdf1m(self, x, **kwargs):
        return self._ilogccdf(_log1mexp(x), **kwargs)

    def _ilogccdf_ilogcdf1m(self, x, **kwargs):
        return self._ilogcdf(_log1mexp(x), **kwargs)

    # Inverse distribution functions via rootfinding
    def _icdf_solve_cdf(self, x, **kwargs):
        a, b = self.support
        res = _solve_bounded(self._cdf, x, min=a, max=b, kwargs=kwargs)
        return res.x

    def _iccdf_solve_ccdf(self, x, **kwargs):
        a, b = self.support
        res = _solve_bounded(self._ccdf, x, min=a, max=b, kwargs=kwargs)
        return res.x

    def _ilogcdf_solve_logcdf(self, x, **kwargs):
        a, b = self.support
        res = _solve_bounded(self._logcdf, x, min=a, max=b, kwargs=kwargs)
        return res.x

    def _ilogccdf_solve_logccdf(self, x, **kwargs):
        a, b = self.support
        res = _solve_bounded(self._logccdf, x, min=a, max=b, kwargs=kwargs)
        return res.x

    @classmethod
    def _draw(cls, sizes=None, rng=None, i_parameterization=0):
        parameterization = cls._parameterizations[i_parameterization]
        shapes = parameterization.draw(sizes, rng)
        return cls(**shapes)


class _Domain:
    symbols = {np.inf: "∞", -np.inf: "-∞", np.pi: "π", -np.pi: "-π"}

    def contains(self, x):
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()


class _SimpleDomain(_Domain):

    def define_parameters(self, *parameters):
        new_symbols = {param.name: param.symbol for param in parameters}
        self.symbols.update(new_symbols)

    def contains(self, item, shapes={}):
        a, b = self.endpoints
        left_inclusive, right_inclusive = self.inclusive

        a = shapes.get(a, a)
        b = shapes.get(b, b)

        in_left = item >= a if left_inclusive else item > a
        in_right = item <= b if right_inclusive else item < b
        return in_left & in_right


class _RealDomain(_SimpleDomain):

    def __init__(self, endpoints=(-oo, oo), inclusive=(False, False)):
        self.endpoints = endpoints
        self.inclusive = inclusive

    def __str__(self):
        a, b = self.endpoints
        left_inclusive, right_inclusive = self.inclusive

        left = "[" if left_inclusive else "("
        a = self.symbols.get(a, f"{a}")
        right = "]" if right_inclusive else ")"
        b = self.symbols.get(b, f"{b}")

        return f"{left}{a}, {b}{right}"


class _IntegerDomain(_SimpleDomain):
    pass


class _Parameter:
    def __init__(self, name, *, symbol, domain, typical):
        self.name = name
        self.symbol = symbol or name
        self.domain = domain
        self.typical = typical

    def __str__(self):
        return f"Accepts `{self.name}` for ${self.symbol} ∈ {str(self.domain)}$."

    def draw(self, size=None, rng=None):
        rng = rng or np.random.default_rng()
        return rng.uniform(*self.typical, size=size)


class _RealParameter(_Parameter):
    def __init__(self, name, *, typical=None, symbol=None, domain=_RealDomain()):
        typical = typical or domain
        symbol = symbol or name
        super().__init__(name, symbol=symbol, domain=domain, typical=typical)

    def check_dtype(self, arr):
        arr = np.asarray(arr)
        dtype = arr.dtype
        valid_dtype = np.ones_like(arr, dtype=bool)
        if np.issubdtype(dtype, np.floating):
            pass
        elif np.issubdtype(dtype, np.integer):
            dtype = np.float64
            arr = np.asarray(arr, dtype=dtype)
        elif np.issubdtype(dtype, np.complexfloating):
            real_arr = np.real(arr)
            valid_dtype = (real_arr == arr)
            arr = real_arr
        else:
            message = f"Parameter {self.name} must be of real dtype."
            raise ValueError(message)
        return arr, valid_dtype

class _IntegerParameter(_Parameter):
    def __init__(self, name, *, typical=None, symbol=None, domain=_IntegerDomain()):
        typical = typical or domain
        symbol = symbol or name
        super().__init__(name, symbol=symbol, domain=domain, typical=typical)

    def check_dtype(self, arr):
        arr = np.asarray(arr)
        dtype = arr.dtype
        valid_dtype = np.ones_like(arr, dtype=bool)
        if np.issubdtype(dtype, np.integer):
            pass
        elif np.issubdtype(dtype, np.inexact):
            integral_arr = np.round(arr)
            valid_dtype = (integral_arr == arr)
            arr = integral_arr
        else:
            message = f"Parameter {self.name} must be of integer dtype."
            raise ValueError(message)
        return arr, valid_dtype


class _Parameterization:
    def __init__(self, *parameters):
        self.parameters = {param.name: param for param in parameters}

    def validate(self, shapes):
        return shapes == set(self.parameters.keys())

    def validate_shapes(self, shapes):
        all_valid = True
        for name, arr in shapes.items():
            parameter = self.parameters[name]
            arr, valid = parameter.check_dtype(arr)
            valid = valid & parameter.domain.contains(arr, shapes)
            all_valid = all_valid & valid
            shapes[name] = arr

        return all_valid

    def __str__(self):
        messages = [str(param) for name, param in self.parameters.items()]
        return " ".join(messages)

    def draw(self, sizes=None, rng=None):
        shapes = {}
        sizes = sizes if np.iterable(sizes) else [sizes]*len(self.parameters)
        for size, param in zip(sizes, self.parameters.values()):
            shapes[param.name] = param.draw(size, rng)
        return shapes


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
    _x_param = _RealParameter('x', domain=_x_support)

    _b_domain.define_parameters(_a_param)
    _log_b_domain.define_parameters(_log_a_param)

    _parameterizations = [_Parameterization(_log_a_param, _log_b_param),
                          _Parameterization(_a_param, _b_param)]
    _variable = _x_param

    def __init__(self, *, a=_null, b=_null, log_a=_null, log_b=_null, **kwargs):
        super().__init__(a=a, b=b, log_a=log_a, log_b=log_b, **kwargs)

    @cached_property
    def a(self):
        return (self._shapes['a'] if 'a' in self._shapes
                else np.exp(self._shapes['log_a']))

    @cached_property
    def b(self):
        return (self._shapes['b'] if 'b' in self._shapes
                else np.exp(self._shapes['log_b']))

    @cached_property
    def log_a(self):
        return (self._shapes['log_a'] if 'log_a' in self._shapes
                else np.log(self._shapes['a']))

    @cached_property
    def log_b(self):
        return (self._shapes['log_b'] if 'log_b' in self._shapes
                else np.log(self._shapes['b']))

    def _logpdf(self, x, *, log_a, log_b, **kwargs):
        return -np.log(x) - np.log(log_b - log_a)
    #
    # def _pdf(self, x, *, log_a, log_b, **kwargs):
    #     return ((log_b - log_a)*x)**-1
    #
    # def _cdf(self, x, *, log_a, log_b, **kwargs):
    #     return (np.log(x) - log_a)/(log_b - log_a)
