import functools
import sys
from functools import cached_property
from scipy._lib._util import _lazywhere
from scipy import special
from scipy.integrate._tanhsinh import _tanhsinh
from scipy.optimize._zeros_py import _chandrupatla, _bracket_root, _differentiate
import numpy as np
_null = object()
oo = np.inf

# TODO:
#  add options for drawing parameters: include endpoints/invalid, log-spacing
#  add `mode` method (can draft without automatic bounding)
#  use `median` information to improve integration?
#  add lower limit to cdf
#  Write `fit` method
#  ensure that user overrides return correct shape and dtype
#  check behavior of moment methods when moments are undefined
#  add `axis` to `ks_1samp`
#  implement symmetric distribution
#  Add loc/scale transformation
#  Add operators for loc/scale transformation
#  Be consistent about options passed to distributions/methods: tols, skip_iv, cache, rng
#  profile/optimize
#  Carefully review input validation, especially for dtype conversions.
#  general cleanup (choose keyword-only parameters)
#  documentation
#  make video
#  add array API support
#  why does dist.ilogcdf(-100) not converge to bound? Check solver response to inf
#  integrate `logmoment` into `moment`? (Not hard, but enough time and code
#   complexity to wait for reviewer feedback before adding.)
#  When there are invalid parameters, icdf fails because _bracket_root_iv
#   raises an error. ValueError: `min <= a < b <= max` must be True...

# Originally, I planned to filter out invalid distribution parameters for the
# author of the distribution; they would always work with "compressed",
# 1D arrays containing only valid distribution parameters. There are two
# problems with this:
# - This essentially requires copying all arrays, even if there is only a
#   single invalid parameter combination. This is expensive. Then, to output
#   the original size data to the user, we need to "decompress" the arrays
#   and fill in the NaNs, so more copying. Unless we branch the code when
#   there are no invalid data, these copies happen even in the normal case,
#   where there are no invalid parameter combinations. We should not incur
#   all this overhead in the normal case.
# - For methods that accept arguments other than distribution parameters, the
#   user will pass in arrays that are broadcastable with the original arrays,
#   not the compressed arrays. This means that this same sort of invalid
#   value detection needs to be repeated every time one of these methods is
#   called.
#   The much simpler solution is to keep the data uncompressed but to replace
#   the invalid parameters and arguments with NaNs (and only if some are
#   invalid). With this approach, the copying happens only if/when it is
#   needed. Most functions involved in stats distribution calculations don't
#   mind NaNs; they just return NaN. The behavior "If x_i is NaN, the result
#   is NaN" is explicit in the array API. So this should be fine.
#   I'm also going to leave the data in the original shape. The reason for this
#   is that the user can process distribution parameters as needed and make
#   them @cached_properties. If we leave all the original shapes alone, the
#   input to functions like `pdf` that accept additional arguments will be
#   broadcastable with these @cached_properties. In most cases, this is
#   completely transparent to the author.
#
#   Another important decision is that the *private* methods must accept
#   the distribution parameters as inputs rather than relying on these
#   cached properties directly (although the public methods typically pass
#   the cached values to the private methods). This is because the elementwise
#   algorithms for quadrature, differentiation, root-finding, and minimization
#   require that the input functions are strictly elementwise in the sense
#   that the value output for a given input element does not depend on the
#   shape of the input or that element's location within the input array.
#   When the computation has converged for an element, it is removed from
#   the computation entirely. The shape of the arrays passed to the
#   function will almost never be broadcastable with the shape of the
#   cached parameter arrays.
#
#   Need to work a bit more on caching. It's not as fast as I'd like it to be.
#   lru_cache for methods that don't accept additional arguments would be
#   great, but it can't easily be turned off by the user. With a custom
#   cache, we can easily add options that disabled or cleared it as needed.
#   Perhaps there is a way to wrap `lru_cache` or the function wrapped by
#   `lru_cache` to add that in.


class _Domain:
    """ Representation of the applicable domain of a parameter or variable

    A `_Domain` object is responsible for storing information about the
    domain of a parameter or variable, determining whether a value is within
    the domain (`contains`), and providing a text/mathematical representation
    of itself (`__str__`). Because the domain of a parameter/variable can have
    a complicated relationship with other parameters and variables of a
    distribution, `_Domain` itself does not try to represent all possibilities;
    in fact, it has no implementation and is meant for subclassing.

    Attributes
    ----------
    symbols : dict
        A map from special numerical values to symbols for use in `__str__`

    Methods
    -------
    contains(x)
        Determine whether the argument is contained within the domain (True)
        or not (False). Used for input validation.
    __str__()
        Returns a text representation of the domain (e.g. `[-π, ∞)`).
        Used for generating documentation.

    """
    symbols = {np.inf: "∞", -np.inf: "-∞", np.pi: "π", -np.pi: "-π"}

    def contains(self, x):
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()


class _SimpleDomain(_Domain):
    """ Representation of a simply-connected domain defined by two endpoints

    Each endpoint may be a finite scalar, positive or negative infinity, or
    be given by a single parameter. The domain may include the endpoints or
    not.

    This class still does not provide an implementation of the __str__ method,
    so it is meant for subclassing (e.g. a subclass for domains on the real
    line).

    Attributes
    ----------
    symbols : dict
        Inherited. A map from special values to symbols for use in `__str__`.
    endpoints : 2-tuple of float(s) and/or str(s)
        A tuple with two values. Each may be either a float (the numerical
        value of the endpoints of the domain) or a string (the name of the
        parameters that will define the endpoint).
    inclusive : 2-tuple of bools
        A tuple with two boolean values; each indicates whether the
        corresponding endpoint is included within the domain or not.

    Methods
    -------
    define_parameters(*parameters)
        Records any parameters used to define the endpoints of the domain
    contains(item, parameter_values)
        Determines whether the argument is contained within the domain

    """

    def define_parameters(self, *parameters):
        r""" Records any parameters used to define the endpoints of the domain

        Adds the keyword name of each parameter and its text representation
        to the  `symbols` attribute as key:value pairs.
        For instance, a parameter may be passed into to a distribution's
        initializer using the keyword `log_a`, and the corresponding
        string representation may be '\log(a)'. To form the text
        representation of the domain for use in documentation, the
        _Domain object needs to map from the keyword name used in the code
        to the string representation.

        Returns None, but updates the `symbols` attribute.

        Parameters
        ----------
        *parameters : _Parameter objects
            Parameters that may define the endpoints of the domain.

        """
        new_symbols = {param.name: param.symbol for param in parameters}
        self.symbols.update(new_symbols)

    def get_numerical_endpoints(self, parameter_values):
        """ Get the numerical values of the domain endpoints

        Domain endpoints may be defined symbolically. This returns numerical
        values of the endpoints given numerical values for any variables.

        Parameters
        ----------
        parameter_values : dict
            A dictionary that maps between string variable names and numerical
            values of parameters, which may define the endpoints.

        Returns
        -------
        a, b : ndarray
            Numerical values of the endpoints

        """
        # TODO: ensure outputs are floats
        a, b = self.endpoints
        # If `a` (`b`) is a string - the name of the parameter that defines
        # the endpoint of the domain - then corresponding numerical values
        # will be found in the `parameter_values` dictionary. Otherwise, it is
        # itself the array of numerical values of the endpoint.
        try:
            a = parameter_values.get(a, a)[()]
            b = parameter_values.get(b, b)[()]
        except TypeError as e:
            message = ("The endpoints of the distribution are defined by "
                       "parameters, but their values were not provided. When "
                       f"using a private method of {self.__class__}, pass "
                       "all required distribution parameters as keyword "
                       "arguments.")
            raise TypeError(message) from e

        return np.broadcast_arrays(a, b)

    def contains(self, item, parameter_values={}):
        """Determine whether the argument is contained within the domain

        Parameters
        ----------
        item : ndarray
            The argument
        parameter_values : dict
            A dictionary that maps between string variable names and numerical
            values of parameters, which may define the endpoints.

        Returns
        -------
        out : bool
            True if `item` is within the domain; False otherwise.

        """
        a, b = self.get_numerical_endpoints(parameter_values)
        left_inclusive, right_inclusive = self.inclusive

        in_left = item >= a if left_inclusive else item > a
        in_right = item <= b if right_inclusive else item < b
        return in_left & in_right


class _RealDomain(_SimpleDomain):
    """ Represents a simply-connected subset of the real line

    Completes the implementation of the `_SimpleDomain` class for simple
    domains on the real line.

    """
    def __init__(self, endpoints=(-oo, oo), inclusive=(False, False)):
        a, b = endpoints
        self.endpoints = np.asarray(a)[()], np.asarray(b)[()]
        self.inclusive = inclusive

    def __str__(self):
        a, b = self.endpoints
        left_inclusive, right_inclusive = self.inclusive

        left = "[" if left_inclusive else "("
        a = self.symbols.get(a, f"{a}")
        right = "]" if right_inclusive else ")"
        b = self.symbols.get(b, f"{b}")

        return f"{left}{a}, {b}{right}"

    def draw(self, size=None, rng=None, proportions=None, parameter_values={}):
        """ Draw random values from the domain

        Parameters
        ----------
        size : tuple of ints
            The shape of the array of valid values to be drawn. For now,
            all values are uniformly sampled, but we should add options for
            picking more challenging values (e.g. including endpoints if the
            domain is inclusive; out-of-bounds values; extreme values).
        rng : np.Generator
            The Generator used for drawing random values.
        proportions : tuple of numbers
            A tuple of four non-negative numbers that indicate the expected
            relative proportion of elements that:

            - are strictly within the domain,
            - are at one of the two endpoints,
            - are strictly outside the domain, and
            - are NaN,

            respectively. Default is (1, 0, 0, 0). The number of elements in each category
            is drawn from the multinomial distribution with `np.prod(size)` as
            the number of trials and `proportions` as the event probabilities.
            The values in `proportions` are automatically normalized to sum to
            1.
        parameter_values : dict
            Map between the names of parameters (that define the endpoints)
            and numerical values (arrays).

        """
        rng = rng or np.random.default_rng()
        proportions = (1, 0, 0, 0) if proportions is None else proportions
        pvals = np.abs(proportions)/np.sum(proportions)

        a, b = self.get_numerical_endpoints(parameter_values)
        min = np.maximum(a, np.finfo(a).min/10) if np.any(np.isinf(a)) else a
        max = np.minimum(b, np.finfo(b).max/10) if np.any(np.isinf(b)) else b

        base_shape = min.shape
        extended_shape = np.broadcast_shapes(size, base_shape)
        n = int(np.prod(extended_shape)/np.prod(base_shape))
        n_in, n_on, n_out, n_nan = rng.multinomial(n, pvals)

        # `min` and `max` can have singleton dimensions that correspond with
        # non-singleton dimensions in `size`. We need to be careful to avoid
        # shuffling results (e.g. a value that was generated for the domain
        # [min[i], max[i]] ends up at index j). To avoid this:
        # - Squeeze the singleton dimensions out of `min`/`max`. Squeezing is
        #   often not the right thing to do, but here is equivalent to moving
        #   all the dimensions that are singleton in `min`/`max` (which may be
        #   non-singleton in the result) to the left. This is what we want.
        # - Now all the non-singleton dimensions of the result are on the left.
        #   Ravel them to a single dimension of length `n`, which is now along
        #   the 0th axis.
        # - Reshape the 0th axis back to the required dimensions, and move
        #   these axes back to their original places.
        base_shape_padded = ((1,)*(len(extended_shape) - len(base_shape))
                             + base_shape)
        base_singletons = np.where(np.asarray(base_shape_padded)==1)[0]
        new_base_singletons = tuple(range(len(base_singletons)))
        # Base singleton dimensions are going to get expanded to these lengths
        shape_expansion = np.asarray(extended_shape)[base_singletons]

        # assert(np.prod(shape_expansion) == n)  # check understanding
        # min = np.reshape(min, base_shape_padded)
        # max = np.reshape(max, base_shape_padded)
        # min = np.moveaxis(min, base_singletons, new_base_singletons)
        # max = np.moveaxis(max, base_singletons, new_base_singletons)
        # squeezed_base_shape = max.shape[len(base_singletons):]
        # assert np.all(min.reshape(squeezed_base_shape) == min.squeeze())
        # assert np.all(max.reshape(squeezed_base_shape) == max.squeeze())

        min = min.squeeze()
        max = max.squeeze()
        squeezed_base_shape = max.shape

        z_in = rng.uniform(min, max, size=(n_in,) + squeezed_base_shape)

        z_on_shape = (n_on,) + squeezed_base_shape
        z_on = np.ones(z_on_shape)
        z_on[:n_on // 2] = min
        z_on[n_on // 2:] = max

        z_out = rng.uniform(min-10, max+10, size=(n_out,) + squeezed_base_shape)

        z_nan = np.full((n_nan,) + squeezed_base_shape, np.nan)

        z = np.concatenate((z_in, z_on, z_out, z_nan), axis=0)
        z = rng.permuted(z, axis=0)

        z = np.reshape(z, tuple(shape_expansion) + squeezed_base_shape)
        z = np.moveaxis(z, new_base_singletons, base_singletons)
        return z


class _IntegerDomain(_SimpleDomain):
    """ Represents a domain of consecutive integers.

    Completes the implementation of the `_SimpleDomain` class for domains
    composed of consecutive integer values.

    To be completed.
    """
    pass


class _Parameter:
    """ Representation of a distribution parameter or variable

    A `_Parameter` object is responsible for storing information about a
    parameter or variable, providing input validation/standardization of
    values passed for that parameter, providing a text/mathematical
    representation of the parameter for the documentation (`__str__`), and
    drawing random values of itself for testing and benchmarking. It does
    not provide a complete implementation of this functionality and is meant
    for subclassing.

    Attributes
    ----------
    name : str
        The keyword used to pass numerical values of the parameter into the
        initializer of the distribution
    symbol : str
        The text representation of the variable in the documentation. May
        include LaTeX.
    domain : _Domain
        The domain of the parameter for which the distribution is valid.
    typical : 2-tuple of floats or strings (consider making a _Domain)
        Defines the endpoints of a typical range of values of the parameter.
        Used for sampling.

   """
    def __init__(self, name, *, domain, symbol=None, typical=None):
        self.name = name
        self.symbol = symbol or name
        self.domain = domain
        if typical is not None and not isinstance(typical, _Domain):
            typical = _RealDomain(typical)
        self.typical = typical or domain

    def __str__(self):
        """ String representation of the parameter for use in documentation """
        return f"Accepts `{self.name}` for ${self.symbol} ∈ {str(self.domain)}$."

    def draw(self, size=None, *, rng=None, domain='typical', proportions=None,
             parameter_values={}):
        """ Draw random values of the parameter for use in testing

        Parameters
        ----------
        size : tuple of ints
            The shape of the array of valid values to be drawn. For now,
            all values are uniformly sampled from within the `typical` range,
            but we should add options for picking more challenging values (e.g.
            including endpoints; out-of-bounds values; extreme values).
        rng : np.Generator
            The Generator used for drawing random values.
        domain : str
            The domain of the `_Parameter` from which to draw. Default is
            "domain" (the *full* domain); alternative is "typical". An
            enhancement would give a way to interpolate between the two.
        proportions : tuple of numbers
            A tuple of four non-negative numbers that indicate the expected
            relative proportion of elements that:

            - are strictly within the domain,
            - are at one of the two endpoints,
            - are strictly outside the domain, and
            - are NaN,

            respectively. Default is (1, 0, 0, 0). The number of elements in each category
            is drawn from the multinomial distribution with `np.prod(size)` as
            the number of trials and `proportions` as the event probabilities.
            The values in `proportions` are automatically normalized to sum to
            1.
        parameter_values : dict
            Map between the names of parameters (that define the endpoints of
            `typical`) and numerical values (arrays).

        """
        domain = getattr(self, domain)
        proportions = (1, 0, 0, 0) if proportions is None else proportions
        return domain.draw(size=size, rng=rng, proportions=proportions,
                           parameter_values=parameter_values)


class _RealParameter(_Parameter):
    """ Represents a real-valued parameter

    Implements the remaining methods of _Parameter for real parameters.
    All attributes are inherited.

    """
    def validate(self, arr, parameter_values):
        """ Input validation/standardization of numerical values of a parameter

        Checks whether elements of the argument `arr` are reals, ensuring that
        the dtype reflects this. Also produces a logical array that indicates
        which elements meet the requirements.

        Parameters
        ----------
        arr : ndarray
            The argument array to be validated and standardized.
        parameter_values : dict
            Map of parameter names to parameter value arrays.

        Returns
        -------
        arr : ndarray
            The argument array that has been validated and standardized
            (converted to an appropriate dtype, if necessary).
        valid_dtype : boolean ndarray
            Logical array indicating which elements are valid reals (True) and
            which are not (False). The arrays of all distribution parameters
            will be broadcasted, and elements for which any parameter value
            does not meet the requirements will be replaced with NaN.

        """
        arr = np.asarray(arr)

        if np.issubdtype(arr.dtype, np.floating):
            valid_dtype = np.ones_like(arr, dtype=bool)
        elif np.issubdtype(arr.dtype, np.integer):
            valid_dtype = np.ones_like(arr, dtype=bool)
            arr = np.asarray(arr, dtype=np.float64)
        elif np.issubdtype(arr.dtype, np.complexfloating):
            real_arr = np.real(arr)
            valid_dtype = (real_arr == arr)
            arr = real_arr
        else:
            message = f"Parameter {self.name} must be of real dtype."
            raise ValueError(message)

        in_domain = self.domain.contains(arr, parameter_values)
        valid = in_domain & valid_dtype

        return arr, arr.dtype, valid


class _Parameterization:
    """ Represents a parameterization of a distribution

    Distributions can have multiple parameterizations. A `_Parameterization`
    object is responsible for recording the parameters used by the
    parameterization, checking whether keyword arguments passed to the
    distribution match the parameterization, and performing input validation
    of the numerical values of these parameters.

    Attributes
    ----------
    parameters : dict
        String names (of keyword arguments) and the corresponding _Parameters.

    """
    def __init__(self, *parameters):
        self.parameters = {param.name: param for param in parameters}

    def __len__(self):
        return len(self.parameters)

    def matches(self, parameters):
        """ Checks whether the keyword arguments match the parameterization

        Parameters
        ----------
        parameters : set
            Set of names of parameters passed into the distribution as keyword
            arguments.

        Returns
        -------
        out : bool
            True if the keyword arguments names match the names of the
            parameters of this parameterization.
        """
        return parameters == set(self.parameters.keys())

    def validation(self, parameter_values):
        """ Input validation / standardization of parameterization

        Parameters
        ----------
        parameter_values : dict
            The keyword arguments passed as parameter values to the
            distribution.

        Returns
        -------
        all_valid : ndarray
            Logical array indicating the elements of the broadcasted arrays
            for which all parameter values are valid.
        dtype : dtype
            The common dtype of the parameter arrays. This will determine
            the dtype of the output of distribution methods.
        """
        all_valid = True
        dtypes = []
        for name, arr in parameter_values.items():
            parameter = self.parameters[name]
            arr, dtype, valid = parameter.validate(arr, parameter_values)
            dtypes.append(dtype)
            all_valid = all_valid & valid
            parameter_values[name] = arr
        dtype = np.result_type(*dtypes)

        return all_valid, dtype

    def __str__(self):
        messages = [str(param) for name, param in self.parameters.items()]
        return " ".join(messages)

    def draw(self, sizes=None, rng=None, proportions=None):
        # ENH: be smart about the order. The domains of some parameters
        # depend on others. If the relationshp is simple (e.g. a < b < c),
        # we could just draw values in order a, b, c.
        parameter_values = {}

        if not len(sizes) or not np.iterable(sizes[0]):
            sizes = [sizes]*len(self.parameters)

        for size, param in zip(sizes, self.parameters.values()):
            parameter_values[param.name] = param.draw(
                size, rng=rng, proportions=proportions,
                parameter_values=parameter_values)

        return parameter_values


def _set_invalid_nan(f):
    # improve structure
    # need to come back to this to think more about use of < vs <=
    # update this
    use_support = {'logpdf', 'logcdf', 'logccdf', 'pdf', 'cdf', 'ccdf'}
    use_01 = {'icdf', 'iccdf'}
    replace_strict = {'pdf', 'logpdf'}
    replace_exact = {'icdf', 'iccdf', 'ilogcdf', 'ilogccdf'}

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

        # Ensure that argument is at least as precise as distribution
        # parameters, which are already at least floats. This will avoid issues
        # with raising integers to negative integer powers failure to replace
        # invalid integers with NaNs.
        dtype = np.result_type(x.dtype, self._dtype)
        if x.dtype != self._dtype:
            x = x.astype(dtype, copy=True)

        # check implications of <, <=
        mask_low = x < low if method_name in replace_strict else x <= low
        mask_high = x > high if method_name in replace_strict else x >= high
        if method_name in replace_exact:
            x, a, b = np.broadcast_arrays(x, *self.support)
            mask_low_exact = (x == low)
            replace_low_exact = b[mask_low_exact] if method_name.endswith('ccdf') else a[mask_low_exact]
            mask_high_exact = (x == high)
            replace_high_exact = a[mask_high_exact] if method_name.endswith('ccdf') else b[mask_high_exact]

        x_invalid = (mask_low | mask_high)
        if np.any(x_invalid):
            x = np.copy(x)
            x[x_invalid] = np.nan
        # TODO: ensure dtype is at least float and that output shape is correct
        # see _set_invalid_nan_property below
        out = np.asarray(f(self, x, *args, **kwargs))
        out[mask_low] = replace_low
        out[mask_high] = replace_high
        if method_name in replace_exact:
            out[mask_low_exact] = replace_low_exact
            out[mask_high_exact] = replace_high_exact

        return out[()]

    return filtered


def _set_invalid_nan_property(f):
    # maybe implement the cache here
    def filtered(self, *args, method=None, skip_iv=False, **kwargs):
        if self.skip_iv or skip_iv:
            return f(self, *args, method=method, **kwargs)

        res = f(self, *args, method=method, **kwargs)
        if res is None:
            # message could be more appropriate
            raise NotImplementedError(self._not_implemented)

        res = np.asarray(res)
        dtype = np.result_type(res.dtype, self._dtype)

        if dtype != self._dtype:  # this won't work for logmoments (complex)
            res = res.astype(dtype, copy=True)

        if res.shape != self._shape:  # faster to check first
            res = np.broadcast_to(res, self._shape)

        # Does the right thing happen without explicitly adding NaNs?
        # if self._any_invalid or dtype != self._dtype:
        #     res = res.astype(dtype, copy=True)
        #     res[self._invalid] = np.nan  # might be redundant

        return res[()]

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


def _log1mexp(x):
    def f1(x):
        return np.log1p(-np.exp(x))

    def f2(x):
        return np.real(np.log(-special.expm1(x + 0j)))

    return _lazywhere(x < -1, (x,), f=f1, f2=f2)


def _logexpxmexpy(x, y):
    # TODO: properly avoid NaN when y is negative infinity
    i = np.isneginf(np.real(y))
    if np.any(i):
        y = y.copy()
        y[i] = np.finfo(y.dtype).min
    x, y = np.broadcast_arrays(x, y)
    return special.logsumexp([x, y+np.pi*1j], axis=0)


def _log_real_standardize(x):
    # log of a negative number has imaginary part that is a multiple of pi*1j,
    # which indicates the sign of the original number. Standardize so that the
    # log of a positive real has 0 imaginary part and the log of a negative
    # real has pi*1j imaginary part.
    shape = x.shape
    x = np.atleast_1d(x)
    real = np.real(x).astype(x.dtype)
    complex = np.imag(x)
    y = real
    negative = np.exp(complex*1j) < 0.5
    y[negative] = y[negative] + np.pi * 1j
    return y.reshape(shape)[()]


class ContinuousDistribution:

    def __init__(self, *, tol=_null, skip_iv=False, **parameters):
        self.tol = tol
        self.skip_iv = skip_iv
        self._moment_raw_cache = {}
        self._moment_central_cache = {}
        self._moment_standard_cache = {}
        parameters = {key: val for key, val in parameters.items()
                      if val is not _null}

        self._not_implemented = (
            f"`{self.__class__.__name__}` does not provide an accurate "
            "implementation of the required method. Leave `tol` unspecified "
            "to use the default implementation."
        )

        if skip_iv or not len(self._parameterizations):
            self._parameters = parameters
            self._invalid = np.asarray(False)  # FIXME: needs the right ndim
            self._any_invalid = False
            self._shape = tuple()
            self._dtype = np.float64
            # Not sure that attributes should be set if we're skipping IV
            # self._set_parameter_attributes()
            return

        parameterization = self._identify_parameterization(parameters)
        parameters, shape = self._broadcast(parameters)
        parameters, invalid, any_invalid, dtype = self._validate(
            parameterization, parameters)

        self._parameters = self._process_parameters(**parameters)
        self._invalid = invalid
        self._any_invalid = any_invalid
        self._shape = shape
        self._dtype = dtype
        self._set_parameter_attributes()

    @classmethod
    def _identify_parameterization(cls, parameters):
        # identify parameterization
        parameter_names_vals = tuple(zip(*parameters.items()))
        parameter_names_vals = parameter_names_vals or ([], [])
        parameter_names, parameter_vals = parameter_names_vals
        parameter_names_set = set(parameter_names)
        for parameterization in cls._parameterizations:
            if parameterization.matches(parameter_names_set):
                break
        else:
            message = (f"The provided parameters `{parameter_names_set}` "
                       "do not match a supported parameterization of the "
                       f"`{cls.__name__}` distribution family.")
            raise ValueError(message)
        return parameterization

    @classmethod
    def _broadcast(cls, parameters):
        # broadcast parameters
        try:
            parameter_vals = np.broadcast_arrays(*parameters.values())
        except ValueError as e:
            message = (f"The parameters {set(parameters)} provided to the "
                       f"`{cls.__name__}` distribution family "
                       "cannot be broadcast to the same shape.")
            raise ValueError(message) from e
        return (dict(zip(parameters.keys(), parameter_vals)),
                parameter_vals[0].shape)

    @classmethod
    def _validate(cls, parameterization, parameters):
        # Replace invalid parameters with `np.nan` and get NaN pattern
        valid, dtype = parameterization.validation(parameters)
        invalid = ~valid
        any_invalid = np.any(invalid)
        # If necessary, make the arrays contiguous and replace invalid with NaN
        if any_invalid:
            for parameter_name in parameters:
                parameters[parameter_name] = np.copy(parameters[parameter_name])
                parameters[parameter_name][invalid] = np.nan

        return parameters, invalid, any_invalid, dtype


    def _set_parameter_attributes(self):
        for name, val in self._parameters.items():
            setattr(self, name, val)


    @classmethod
    def _draw(cls, sizes=None, rng=None, i_parameterization=None,
              proportions=None):
        if len(cls._parameterizations) == 0:
            return cls()
        if i_parameterization is None:
            n = cls._num_parameterizations()
            i_parameterization = rng.integers(0, max(0, n - 1), endpoint=True)

        parameterization = cls._parameterizations[i_parameterization]
        parameters = parameterization.draw(sizes, rng, proportions=proportions)
        return cls(**parameters)

    @classmethod
    def _num_parameterizations(cls):
        return len(cls._parameterizations)

    @classmethod
    def _num_parameters(cls):
        return (0 if not cls._num_parameterizations()
                else len(cls._parameterizations[0]))

    def _overrides(self, method_name):
        method = getattr(self.__class__, method_name, None)
        super_method = getattr(self.__class__.__base__, method_name, None)
        return method is not super_method

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
        return self._support(**self._parameters)

    def _support(self, **kwargs):
        return self._variable.domain.get_numerical_endpoints(kwargs)

    def logentropy(self, *, method=None):
        return self._logentropy_dispatch(method=method, **self._parameters)

    def _logentropy_dispatch(self, method=None, **kwargs):
        if method in {None, 'formula'} and self._overrides('_logentropy'):
            return self._logentropy(**kwargs)
        elif (self.tol is _null and self._overrides('_entropy') and method is None) or method=='log/exp':
            return self._logentropy_log_entropy(**kwargs)
        elif method in {'quadrature', None}:
            return self._logentropy_integrate_logpdf(**kwargs)
        else:
            raise NotImplementedError(self._not_implemented)

    def _logentropy_log_entropy(self, **kwargs):
        res = np.log(self._entropy_dispatch(**kwargs) + 0j)
        return _log_real_standardize(res)

    def _logentropy_integrate_logpdf(self, **kwargs):
        def logintegrand(x, **kwargs):
            logpdf = self._logpdf_dispatch(x, **kwargs)
            return logpdf + np.log(0j+logpdf)
        res = self._quadrature(logintegrand, kwargs=kwargs, log=True)
        return _log_real_standardize(res + np.pi*1j)

    def entropy(self, *, method=None):
        return self._entropy_dispatch(method=method, **self._parameters)

    def _entropy_dispatch(self, method=None, **kwargs):
        if method in {None, 'formula'} and self._overrides('_entropy'):
            return self._entropy(**kwargs)
        elif (self._overrides('_logentropy') and method is None) or method=='log/exp':
            return self._entropy_exp_logentropy(**kwargs)
        elif method in {'quadrature', None}:
            return self._entropy_integrate_pdf(**kwargs)
        else:
            raise NotImplementedError(self._not_implemented)

    def _entropy_exp_logentropy(self, **kwargs):
        return np.exp(self._logentropy_dispatch(**kwargs))

    def _entropy_integrate_pdf(self, **kwargs):
        def integrand(x, **kwargs):
            pdf = self._pdf_dispatch(x, **kwargs)
            return np.log(pdf)*pdf
        return -self._quadrature(integrand, kwargs=kwargs)

    def median(self, *, method=None):
        return self._median_dispatch(method=method, **self._parameters)

    def _median_dispatch(self, method=None, **kwargs):
        if method in {None, 'formula'} and self._overrides('_median'):
            return np.asarray(self._median(**kwargs))[()]
        elif method in {None, 'icdf'}:
            return self._median_icdf(**kwargs)
        else:
            raise NotImplementedError(self._not_implemented)

    def _median_icdf(self, **kwargs):
        return self._icdf_dispatch(0.5, **kwargs)

    @_set_invalid_nan_property
    def mean(self, *, method=None):
        methods = {method} if method is not None else self._moment_methods
        return self._moment_raw_dispatch(1, methods=methods, **self._parameters)

    @_set_invalid_nan_property
    def var(self, *, method=None):
        methods = {method} if method is not None else self._moment_methods
        return self._moment_central_dispatch(2, methods=methods, **self._parameters)

    def std(self, *, method=None):
        return np.sqrt(self.var(method=method))

    @_set_invalid_nan_property
    def skewness(self, *, method=None):
        methods = {method} if method is not None else self._moment_methods
        return self._moment_standard_dispatch(3, methods=methods, **self._parameters)

    @_set_invalid_nan_property
    def kurtosis(self, *, method=None):
        methods = {method} if method is not None else self._moment_methods
        return self._moment_standard_dispatch(4, methods=methods, **self._parameters)

    @_set_invalid_nan
    def logpdf(self, x, *, method=None):
        return self._logpdf_dispatch(x, method=method, **self._parameters)

    def _logpdf_dispatch(self, x, *, method=None, **kwargs):
        if method in {None, 'formula'} and self._overrides('_logpdf'):
            return self._logpdf(x, **kwargs)
        elif (self.tol is _null and method is None) or method == 'log/exp':
            return self._logpdf_log_pdf(x, **kwargs)
        else:
            raise NotImplementedError(self._not_implemented)

    def _logpdf_log_pdf(self, x, **kwargs):
        return np.log(self._pdf_dispatch(x, **kwargs))

    @_set_invalid_nan
    def pdf(self, x, *, method=None):
        return self._pdf_dispatch(x, method=method, **self._parameters)

    def _pdf_dispatch(self, x, *, method=None, **kwargs):
        if method in {None, 'formula'} and self._overrides('_pdf'):
            return self._pdf(x, **kwargs)
        if (self._overrides('_logpdf') and method is None) or method == 'log/exp':
            return self._pdf_exp_logpdf(x, **kwargs)
        else:
            raise NotImplementedError(self._not_implemented)

    def _pdf_exp_logpdf(self, x, **kwargs):
        return np.exp(self._logpdf_dispatch(x, **kwargs))

    @_set_invalid_nan
    def logcdf(self, x, *, method=None):
        return self._logcdf_dispatch(x, method=method, **self._parameters)

    def _logcdf_dispatch(self, x, *, method=None, **kwargs):
        if method in {None, 'formula'} and self._overrides('_logcdf'):
            return self._logcdf(x, **kwargs)
        elif (self.tol is _null and self._overrides('_cdf') and method is None) or method=='log/exp':
            return self._logcdf_log_cdf(x, **kwargs)
        elif (self._overrides('_logccdf') and method is None) or method=='complementarity':
            return self._logcdf_log1mexpccdf(x, **kwargs)
        elif method in {'quadrature', None}:
            return self._logcdf_integrate_logpdf(x, **kwargs)
        else:
            raise NotImplementedError(self._not_implemented)

    def _logcdf_log_cdf(self, x, **kwargs):
        return np.log(self._cdf_dispatch(x, **kwargs))

    def _logcdf_log1mexpccdf(self, x, **kwargs):
        return _log1mexp(self._logccdf_dispatch(x, **kwargs))


    def _logcdf_integrate_logpdf(self, x, **kwargs):
        a, _ = self._support(**kwargs)
        return self._quadrature(self._logpdf_dispatch, limits=(a, x),
                                kwargs=kwargs, log=True)

    @_set_invalid_nan
    def cdf(self, x, *, method=None):
        return self._cdf_dispatch(x, method=method, **self._parameters)

    def _cdf_dispatch(self, x, *, method=None, **kwargs):
        if method in {None, 'formula'} and self._overrides('_cdf'):
            return self._cdf(x, **kwargs)
        elif (self._overrides('_logcdf') and method is None) or method=='log/exp':
            return self._cdf_exp_logcdf(x, **kwargs)
        elif (self._tol is _null and self._overrides('_ccdf') and method is None) or method=='complementarity':
            return self._cdf_1mccdf(x, **kwargs)
        elif method in {'quadrature', None}:
            return self._cdf_integrate_pdf(x, **kwargs)
        else:
            raise NotImplementedError(self._not_implemented)

    def _cdf_exp_logcdf(self, x, **kwargs):
        return np.exp(self._logcdf_dispatch(x, **kwargs))

    def _cdf_1mccdf(self, x, **kwargs):
        return 1 - self._ccdf_dispatch(x, **kwargs)

    def _cdf_integrate_pdf(self, x, **kwargs):
        a, _ = self._support(**kwargs)
        return self._quadrature(self._pdf_dispatch, limits=(a, x),
                                kwargs=kwargs)

    @_set_invalid_nan
    def logccdf(self, x, *, method=None):
        return self._logccdf_dispatch(x, method=method, **self._parameters)

    def _logccdf_dispatch(self, x, method=None, **kwargs):
        if method in {None, 'formula'} and self._overrides('_logccdf'):
            return self._logccdf(x, **kwargs)
        if (self.tol is _null and self._overrides('_cdf') and method is None) or method=='log/exp':
            return self._logccdf_log_ccdf(x, **kwargs)
        elif (self._overrides('_logcdf') and method is None) or method=='complementarity':
            return self._logccdf_log1mexpcdf(x, **kwargs)
        elif method in {'quadrature', None}:
            return self._logccdf_integrate_logpdf(x, **kwargs)
        else:
            raise NotImplementedError(self._not_implemented)

    def _logccdf_log_ccdf(self, x, **kwargs):
        return np.log(self._ccdf_dispatch(x, **kwargs))

    def _logccdf_log1mexpcdf(self, x, **kwargs):
        return _log1mexp(self._logcdf_dispatch(x, **kwargs))

    def _logccdf_integrate_logpdf(self, x, **kwargs):
        _, b = self._support(**kwargs)
        return self._quadrature(self._logpdf_dispatch, limits=(x, b),
                                kwargs=kwargs, log=True)

    @_set_invalid_nan
    def ccdf(self, x, *, method=None):
        return self._ccdf_dispatch(x, method=method, **self._parameters)

    def _ccdf_dispatch(self, x, method=None, **kwargs):
        if method in {None, 'formula'} and self._overrides('_ccdf'):
            return self._ccdf(x, **kwargs)
        elif (self._overrides('_logccdf') and method is None) or method=='log/exp':
            return self._ccdf_exp_logccdf(x, **kwargs)
        elif (self._tol is _null and self._overrides('_cdf') and method is None) or method=='complementarity':
            return self._ccdf_1mcdf(x, **kwargs)
        elif method in {'quadrature', None}:
            return self._ccdf_integrate_pdf(x, **kwargs)
        else:
            raise NotImplementedError(self._not_implemented)

    def _ccdf_exp_logccdf(self, x, **kwargs):
        return np.exp(self._logccdf_dispatch(x, **kwargs))

    def _ccdf_1mcdf(self, x, **kwargs):
        return 1 - self._cdf_dispatch(x, **kwargs)

    def _ccdf_integrate_pdf(self, x, **kwargs):
        _, b = self._support(**kwargs)
        return self._quadrature(self._pdf_dispatch, limits=(x, b),
                                kwargs=kwargs)

    @_set_invalid_nan
    def ilogcdf(self, x, *, method=None):
        return self._ilogcdf_dispatch(x, method=method, **self._parameters)

    def _ilogcdf_dispatch(self, x, method=None, **kwargs):
        if method in {None, 'formula'} and self._overrides('_ilogcdf'):
            return self._ilogcdf(x, **kwargs)
        elif (self._overrides('_ilogccdf') and method is None) or method=='complementarity':
            return self._ilogcdf_ilogccdf1m(x, **kwargs)
        elif method in {None, 'inversion'}:
            return self._ilogcdf_solve_logcdf(x, **kwargs)
        else:
            raise NotImplementedError(self._not_implemented)

    def _ilogcdf_ilogccdf1m(self, x, **kwargs):
        return self._ilogccdf_dispatch(_log1mexp(x), **kwargs)

    def _ilogcdf_solve_logcdf(self, x, **kwargs):
        return self._solve_bounded(self._logcdf_dispatch, x, kwargs=kwargs)

    @_set_invalid_nan
    def icdf(self, x, *, method=None):
        return self._icdf_dispatch(x, method=method, **self._parameters)

    def _icdf_dispatch(self, x, method=None, **kwargs):
        if method in {None, 'formula'} and self._overrides('_icdf'):
            return self._icdf(x, **kwargs)
        elif (self.tol is _null and self._overrides('_iccdf') and method is None) or method=='complementarity':
            return self._icdf_iccdf1m(x, **kwargs)
        elif method in {None, 'inversion'}:
            return self._icdf_solve_cdf(x, **kwargs)
        else:
            raise NotImplementedError(self._not_implemented)

    def _icdf_iccdf1m(self, x, **kwargs):
        return self._iccdf_dispatch(1 - x, **kwargs)

    def _icdf_solve_cdf(self, x, **kwargs):
        return self._solve_bounded(self._cdf_dispatch, x, kwargs=kwargs)

    @_set_invalid_nan
    def ilogccdf(self, x, *, method=None):
        return self._ilogccdf_dispatch(x, method=method, **self._parameters)

    def _ilogccdf_dispatch(self, x, method=None, **kwargs):
        if method in {None, 'formula'} and self._overrides('_ilogccdf'):
            return self._ilogccdf(x, **kwargs)
        elif (self._overrides('_ilogcdf') and method is None) or method=='complementarity':
            return self._ilogccdf_ilogcdf1m(x, **kwargs)
        elif method in {None, 'inversion'}:
            return self._ilogccdf_solve_logccdf(x, **kwargs)
        else:
            raise NotImplementedError(self._not_implemented)

    def _ilogccdf_ilogcdf1m(self, x, **kwargs):
        return self._ilogcdf_dispatch(_log1mexp(x), **kwargs)

    def _ilogccdf_solve_logccdf(self, x, **kwargs):
        return self._solve_bounded(self._logccdf_dispatch, x, kwargs=kwargs)

    @_set_invalid_nan
    def iccdf(self, x, *, method=None):
        return self._iccdf_dispatch(x, method=method, **self._parameters)

    def _iccdf_dispatch(self, x, method=None, **kwargs):
        if method in {None, 'formula'} and self._overrides('_iccdf'):
            return self._iccdf(x, **kwargs)
        elif (self.tol is _null and self._overrides('_icdf') and method is None) or method=='complementarity':
            return self._iccdf_icdf1m(x, **kwargs)
        elif method in {None, 'inversion'}:
            return self._iccdf_solve_ccdf(x, **kwargs)
        else:
            raise NotImplementedError(self._not_implemented)

    def _iccdf_icdf1m(self, x, **kwargs):
        return self._icdf_dispatch(1 - x, **kwargs)

    def _iccdf_solve_ccdf(self, x, **kwargs):
        return self._solve_bounded(self._ccdf_dispatch, x, kwargs=kwargs)

    def sample(self, shape=(), *, rng=None):
        shape = (shape,) if not np.iterable(shape) else tuple(shape)
        rng = np.random.default_rng() if rng is None else rng
        return self._sample(shape, rng, **self._parameters)

    def _sample(self, shape, rng, **kwargs):
        full_shape = shape + self._shape
        uniform = rng.uniform(size=full_shape)
        return self._icdf_dispatch(uniform, **kwargs)

    def _logmoment(self, order=1, *, logcenter=None, standardized=False):
        # make this private until it is worked into moment
        if logcenter is None or standardized is True:
            logmean = self._logmoment_quad(1, -np.inf, **self._parameters)
        else:
            logmean = None

        logcenter = logmean if logcenter is None else logcenter
        res = self._logmoment_quad(order, logcenter, **self._parameters)
        if standardized:
            logvar = self._logmoment_quad(2, logmean, **self._parameters)
            res = res - logvar * (order/2)
        return res

    def _logmoment_quad(self, order, logcenter, **kwargs):
        def logintegrand(x, order, logcenter, **kwargs):
            logpdf = self._logpdf_dispatch(x, **kwargs)
            return logpdf + order*_logexpxmexpy(np.log(x+0j), logcenter)
        return self._quadrature(logintegrand, args=(order, logcenter),
                                kwargs=kwargs, log=True)

    def _validate_order(self, order, skip_iv = False):
        if self.skip_iv or skip_iv:
            return order

        order = np.asarray(order, dtype=self._dtype)[()]
        order_int = np.round(order)
        if (order_int.size != 1 or order_int != order
                or order < 0 or not np.isfinite(order)):
            message = '`order` must be a positive, finite integer.'
            raise ValueError(message)
        return order_int

    @cached_property
    def _moment_methods(self):
        return {'cache', 'formula', 'transform',
                'normalize', 'general', 'quadrature'}

    @_set_invalid_nan_property
    def moment_raw(self, order=1, *, method=None):
        order = self._validate_order(order)
        methods = self._moment_methods if method is None else {method}
        return self._moment_raw_dispatch(order, methods=methods,
                                         **self._parameters)

    def _moment_raw_dispatch(self, order, *, methods, **kwargs):
        # How to indicate to the user if the requested methods could not be used?
        # Rather than returning None when a moment is not available, raise?
        moment = None

        if 'cache' in methods:
            moment = self._moment_raw_cache.get(order, None)

        if moment is None and 'formula' in methods:
            moment = self._moment_raw(order, **kwargs)
            if moment is not None and getattr(moment, 'shape', None) != self._shape:
                moment = np.broadcast_to(moment, self._shape)

        if moment is None and 'transform' in methods and order > 1:
            moment = self._moment_raw_transform(order, **kwargs)

        if moment is None and 'general' in methods:
            moment = self._moment_raw_general(order, **kwargs)

        if moment is None and 'quadrature' in methods:
            moment = self._moment_integrate_pdf(order, center=0, **kwargs)

        if moment is not None:
            self._moment_raw_cache[order] = moment

        return moment

    def _moment_raw(self, order, **kwargs):
        return None

    def _moment_raw_transform(self, order, **kwargs):
        # Doesn't make sense to get the mean by "transform", since that's
        # how we got here. Questionable whether 'quadrature' should be here.

        central_moments = []
        for i in range(int(order) + 1):
            methods = {'cache', 'formula', 'normalize', 'general'}
            moment_i = self._moment_central_dispatch(order=i,
                                                     methods=methods, **kwargs)
            if moment_i is None:
                return None
            central_moments.append(moment_i)

        mean_methods = {'cache', 'formula', 'quadrature'}
        mean = self._moment_raw_dispatch(1, methods=mean_methods, **kwargs)
        if mean is None:
            return None

        moment = self._moment_transform_center(order, central_moments, mean, 0)
        return moment

    def _moment_raw_general(self, order, **kwargs):
        # This is the only general formula for a raw moment of a probability
        # distribution
        return 1 if order == 0 else None

    @_set_invalid_nan_property
    def moment_central(self, order=1, *, method=None):
        order = self._validate_order(order)
        methods = self._moment_methods if method is None else {method}
        return self._moment_central_dispatch(order, methods=methods,
                                             **self._parameters)

    def _moment_central_dispatch(self, order, *, methods, **kwargs):
        moment = None

        if 'cache' in methods:
            moment = self._moment_central_cache.get(order, None)

        if moment is None and 'formula' in methods:
            moment = self._moment_central(order, **kwargs)
            if moment is not None and getattr(moment, 'shape', None) != self._shape:
                moment = np.broadcast_to(moment, self._shape)

        if moment is None and 'transform' in methods:
            moment = self._moment_central_transform(order, **kwargs)

        if moment is None and 'normalize' in methods and order > 2:
            moment = self._moment_central_normalize(order, **kwargs)

        if moment is None and 'general' in methods:
            moment = self._moment_central_general(order, **kwargs)

        if moment is None and 'quadrature' in methods:
            mean = self._moment_raw_dispatch(1, **kwargs,
                                             methods=self._moment_methods)
            moment = self._moment_integrate_pdf(order, center=mean, **kwargs)

        if moment is not None:
            self._moment_central_cache[order] = moment

        return moment

    def _moment_central(self, order, **kwargs):
        return None

    def _moment_central_transform(self, order, **kwargs):

        raw_moments = []
        for i in range(int(order) + 1):
            methods = {'cache', 'formula', 'general'}
            moment_i = self._moment_raw_dispatch(order=i, methods=methods,
                                                 **kwargs)
            if moment_i is None:
                return None
            raw_moments.append(moment_i)

        mean_methods = self._moment_methods
        mean = self._moment_raw_dispatch(1, methods=mean_methods, **kwargs)

        moment = self._moment_transform_center(order, raw_moments, 0, mean)
        return moment

    def _moment_central_normalize(self, order, **kwargs):
        methods = {'cache', 'formula', 'general'}
        standard_moment = self._moment_standard_dispatch(order, **kwargs,
                                                         methods=methods)
        if standard_moment is None:
            return None
        var = self._moment_central_dispatch(2, methods=self._moment_methods,
                                            **kwargs)
        return standard_moment*var**(order/2)

    def _moment_central_general(self, order, **kwargs):
        general_central_moments = {0: 1, 1: 0}
        return general_central_moments.get(order, None)

    @_set_invalid_nan_property
    def moment_standard(self, order=1, *, method=None):
        order = self._validate_order(order)
        methods = self._moment_methods if method is None else {method}
        return self._moment_standard_dispatch(order, methods=methods,
                                              **self._parameters)

    def _moment_standard_dispatch(self, order, *, methods, **kwargs):
        moment = None

        if 'cache' in methods:
            moment = self._moment_standard_cache.get(order, None)

        if moment is None and 'formula' in methods:
            moment = self._moment_standard(order, **kwargs)
            if moment is not None and getattr(moment, 'shape', None) != self._shape:
                moment = np.broadcast_to(moment, self._shape)

        if moment is None and 'normalize' in methods:
            moment = self._moment_standard_transform(order, False, **kwargs)

        if moment is None and 'general' in methods:
            moment = self._moment_standard_general(order, **kwargs)

        if moment is None and 'normalize' in methods:
            moment = self._moment_standard_transform(order, True, **kwargs)

        if moment is not None:
            self._moment_standard_cache[order] = moment

        return moment

    def _moment_standard(self, order, **kwargs):
        return None

    def _moment_standard_transform(self, order, use_quadrature, **kwargs):
        methods = ({'quadrature'} if use_quadrature
                   else {'cache', 'formula', 'transform'})
        central_moment = self._moment_central_dispatch(order, **kwargs,
                                                       methods=methods)
        if central_moment is None:
            return None
        var = self._moment_central_dispatch(2, methods=self._moment_methods,
                                            **kwargs)
        return central_moment/var**(order/2)

    def _moment_standard_general(self, order, **kwargs):
        general_standard_moments = {0: 1, 1: 0, 2: 1}
        return general_standard_moments.get(order, None)

    def _moment_integrate_pdf(self, order, center, **kwargs):
        def integrand(x, order, center, **kwargs):
            pdf = self._pdf_dispatch(x, **kwargs)
            return pdf*(x-center)**order
        return self._quadrature(integrand, args=(order, center), kwargs=kwargs)

    def _moment_transform_center(self, order, moment_as, a, b):
        a, b, *moment_as = np.broadcast_arrays(a, b, *moment_as)
        n = order
        i = np.arange(n+1).reshape([-1]+[1]*a.ndim)  # orthogonal to other axes
        n_choose_i = special.binom(n, i)
        moment_b = np.sum(n_choose_i*moment_as*(a-b)**(n-i), axis=0)
        return moment_b

    def _quadrature(self, integrand, limits=None, args=None, kwargs=None, log=False):
        a, b = self._support(**kwargs) if limits is None else limits
        args = [] if args is None else args
        kwargs = {} if kwargs is None else kwargs
        f, args = kwargs2args(integrand, args=args, kwargs=kwargs)
        res = _tanhsinh(f, a, b, args=args, log=log)
        return res.integral

    def _solve_bounded(self, f, p, *, bounds=None, kwargs=None):
        # should modify _bracket_root and _chandrupatla so we don't need all this
        min, max = self._support(**kwargs) if bounds is None else bounds
        kwargs = {} if kwargs is None else kwargs

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
        return _chandrupatla(f3, a=res.xl, b=res.xr, args=args).x

    ## Easiest to do right now
    # @classmethod
    # def llf(cls, parameters, *, sample, axis=-1):
    #     dist = cls(**parameters)
    #     return np.sum(dist.logpdf(sample), axis=axis)

    ## Works if user doesn't pass in parameters, or passes in the right parameters
    # def llf(self, parameters=None, *, sample, axis=-1):
    #     parameters = self._parameters if parameters is None else parameters
    #     return np.sum(self._logpdf_dispatch(sample, **parameters))

    ## Probably best, but we still need something to process parameters
    ## I have been thinking that I should break out the part of __init__
    ## that identifies the parameterization and validates parameters and
    ## call that here.
    @classmethod
    def llf(cls, parameters, *, sample, axis=-1):
        # still needs input validation of sample
        parameterization = cls._identify_parameterization(parameters)
        parameters, _ = cls._broadcast(parameters)
        parameters, _, _, _ = cls._validate(parameterization, parameters)
        parameters = cls._process_parameters(**parameters)

        return cls._llf(parameters, sample=sample, axis=axis)

    @classmethod
    def _llf(cls, parameters, *, sample, axis):
        logpdf = getattr(cls, '_logpdf', lambda cls, sample, **parameters: (
            np.log(cls._pdf(cls, sample, **parameters))))
        return np.sum(logpdf(cls, sample, **parameters), axis=axis)

    @classmethod
    def dllf(cls, parameters, *, sample, var):
        # relies on current behavior of `_differentiate` to get shapes right
        parameters = parameters.copy()
        parameterization = cls._identify_parameterization(parameters)
        parameters, _ = cls._broadcast(parameters)
        parameters, _, _, _ = cls._validate(parameterization, parameters)

        def f(x):
            params = parameters.copy()
            params[var] = x
            params = cls._process_parameters(**params)
            res = cls._llf(params, sample=sample[:, None], axis=0)
            return np.reshape(res, x.shape)

        return _differentiate(f, parameters[var]).df
