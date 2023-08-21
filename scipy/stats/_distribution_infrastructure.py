import functools
import sys
import enum
from functools import cached_property
from scipy._lib._util import _lazywhere
from scipy import special, optimize
from scipy.integrate._tanhsinh import _tanhsinh
from scipy.optimize._zeros_py import (_chandrupatla, _bracket_root,
                                      _differentiate)
from scipy.optimize._chandrupatla import _chandrupatla_minimize
from scipy.stats.qmc import Halton
import numpy as np
_null = object()
oo = np.inf

# Could add other policies for broadcasting and edge/out-of-bounds case handling
# For instance, when edge case handling is known not to be needed, it's much
# faster to turn it off, but it might still be nice to have array conversion
# and shaping done so the user doesn't need to be so carefuly.
IV_POLICY = enum.Enum('IV_POLICY', ['SKIP_ALL'])

# Alternative to enums is max cache size per function like lru_cache
# Currently, this is only used by `moment_...` methods, but it could be used
# for all methods that don't require an argument (e.g. entropy)
CACHE_POLICY = enum.Enum('CACHE_POLICY', ['NO_CACHE', 'CACHE'])

# TODO:
#  baseline 2-arg cdf
#  figure out dtypes of log methods
#  _parameterizations cannot be a class variable if it is modified
#  test loc/scale distribution
#  address name collisions in transformed distributions
#  Add bounds to `fit` method
#  implement symmetric distribution
#  implement composite distribution
#  check behavior of moment methods when moments are undefined/infinite
#  Be consistent about options passed to distributions/methods: tols, skip_iv,
#    cache, rng. Also check for issues with transformed distributions.
#  profile/optimize
#  general cleanup (choose keyword-only parameters)
#  documentation
#  compare old/new distribution timing
#  make video
#  PR
#  add array API support
#  why does dist.ilogcdf(-100) not converge to bound? Check solver response to inf
#  _chandrupatla_minimize should not report xm = fm = NaN when it fails
#  improve mode after writing _bracket_minimize
#  integrate `logmoment` into `moment`? (Not hard, but enough time and code
#   complexity to wait for reviewer feedback before adding.)
#  Eliminate bracket_root error "`min <= a < b <= max` must be True"
#  Fully-bake addition of lower limit to CDF. It's really sloppy right now.
#   Needs input validation, better method names, better style, and better
#   efficiency. Similar idea needed in `logcdf`.
#  When drawing endpoint/out-of-bounds values of a parameter, draw them from
#   the endpoints/out-of-bounds region of the full `domain`, not `typical`.
#   Make tolerance override method-specific again.
#  Test repr?
#  Fix _scalar_optimization_algorithms with 0-size arrays
#  use `median` information to improve integration? In some cases this will
#   speed things up. If it's not needed, it may be about twice as slow. I think
#   it should depend on the accuracy setting.
#  in tests, check reference value against that produced using np.vectorize?
#  add `axis` to `ks_1samp`
#  Getting `default_rng` takes forever! OK to do it only when support is called?
#  User tips for faster execution:
#  - pass NumPy arrays
#  - pass inputs of floating point type (not integers)
#  - prefer NumPy scalars or 0d arrays over other size 1 arrays
#  - pass no invalid parameters and disable invalid parameter checks with iv_profile
#  - provide a Generator if you're going to do sampling
#  add options for drawing parameters: log-spacing
#  accuracy benchmark suite
#  Should caches be attributes so we can more easily ensure that they are not
#   modified when caching is turned off?
#  Make ShiftedScaledDistribution more efficient - only process underlying
#   distribution parameters as necessary.

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
#
#   I've sprinkled in some optimizations for scalars and same-shape/type arrays
#   throughout. The biggest time sinks before were:
#   - broadcast_arrays
#   - result_dtype
#   - is_subdtype
#   It is much faster to check whether these are necessary than to do them.


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
            a = np.asarray(parameter_values.get(a, a))
            b = np.asarray(parameter_values.get(b, b))
        except TypeError as e:
            message = ("The endpoints of the distribution are defined by "
                       "parameters, but their values were not provided. When "
                       f"using a private method of {self.__class__}, pass "
                       "all required distribution parameters as keyword "
                       "arguments.")
            raise TypeError(message) from e

        return a, b

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
        a, b = np.broadcast_arrays(a, b)
        min = np.maximum(a, np.finfo(a).min/10) if np.any(np.isinf(a)) else a
        max = np.minimum(b, np.finfo(b).max/10) if np.any(np.isinf(b)) else b

        base_shape = min.shape
        extended_shape = np.broadcast_shapes(size, base_shape)
        n_extended = np.prod(extended_shape)
        n_base = np.prod(base_shape)
        n = int(n_extended / n_base) if n_extended else 0

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

        # get copies of min and max with no nans so that uniform doesn't fail
        min_nn, max_nn = min.copy(), max.copy()
        i = np.isnan(min_nn) | np.isnan(max_nn)
        min_nn[i] = 0
        max_nn[i] = 1
        z_in = rng.uniform(min_nn, max_nn, size=(n_in,) + squeezed_base_shape)

        z_on_shape = (n_on,) + squeezed_base_shape
        z_on = np.ones(z_on_shape)
        z_on[:n_on // 2] = min
        z_on[n_on // 2:] = max

        z_out = rng.uniform(min_nn-10, max_nn+10,
                            size=(n_out,) + squeezed_base_shape)

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

        valid_dtype = None
        # minor optimization - fast track the most common types to avoid
        # overhead of np.issubdtype. Checking for `in {...}` doesn't work : /
        if arr.dtype == np.float64 or arr.dtype == np.float32:
            pass
        elif arr.dtype == np.int32 or arr.dtype == np.int64:
            arr = np.asarray(arr, dtype=np.float64)
        elif np.issubdtype(arr.dtype, np.floating):
            pass
        elif np.issubdtype(arr.dtype, np.integer):
            arr = np.asarray(arr, dtype=np.float64)
        elif np.issubdtype(arr.dtype, np.complexfloating):
            real_arr = np.real(arr)
            valid_dtype = (real_arr == arr)
            arr = real_arr
        else:
            message = f"Parameter `{self.name}` must be of real dtype."
            raise ValueError(message)

        valid = self.domain.contains(arr, parameter_values)
        valid = valid & valid_dtype if valid_dtype is not None else valid

        return arr[()], arr.dtype, valid


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
        dtypes = set()  # avoid np.result_type if there's only one type
        for name, arr in parameter_values.items():
            parameter = self.parameters[name]
            arr, dtype, valid = parameter.validate(arr, parameter_values)
            dtypes.add(dtype)
            all_valid = all_valid & valid
            parameter_values[name] = arr
        dtype = arr.dtype if len(dtypes)==1 else np.result_type(*list(dtypes))

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
    endpoints = {'icdf': (0, 1), 'iccdf': (0, 1),
                 'ilogcdf': (-np.inf, 0), 'ilogccdf': (-np.inf, 0)}
    replacements = {'logpdf': (-oo, -oo), 'pdf': (0, 0),
                    'logcdf': (-oo, 0), 'logccdf': (0, -oo),
                    'cdf': (0, 1), 'ccdf': (1, 0), '_cdf1': (0, 1)}
    replace_strict = {'pdf', 'logpdf'}
    replace_exact = {'icdf', 'iccdf', 'ilogcdf', 'ilogccdf'}

    def filtered(self, x, *args, iv_policy=None, **kwargs):
        if (self.iv_policy or iv_policy) == IV_POLICY.SKIP_ALL:
            return f(self, x, *args, **kwargs)

        method_name = f.__name__
        x = np.asarray(x)
        dtype = self._dtype
        shape = self._shape

        # Ensure that argument is at least as precise as distribution
        # parameters, which are already at least floats. This will avoid issues
        # with raising integers to negative integer powers failure to replace
        # invalid integers with NaNs.
        if x.dtype != dtype:
            dtype = np.result_type(x.dtype, dtype)
            x = np.asarray(x, dtype=dtype)

        # Broadcasting is slow. Skip if possible.
        if not x.shape == shape:
            try:
                shape = np.broadcast_shapes(x.shape, shape)
                x = np.broadcast_to(x, shape)
                # Should we broadcast the distribution parameters to match shape of x?
                # Should we copy if we broadcast to avoid passing a view to developer functions?
            except ValueError as e:
                message = (
                    f"The argument provided to `{self.__class__.__name__}"
                    f".{method_name}` cannot be be broadcast to the same "
                    "shape as the distribution parameters.")
                raise ValueError(message) from e

        low, high = endpoints.get(method_name, self.support())
        mask_low = x < low if method_name in replace_strict else x <= low
        mask_high = x > high if method_name in replace_strict else x >= high
        mask_invalid = (mask_low | mask_high)
        any_invalid = (mask_invalid if mask_invalid.shape == ()
                       else np.any(mask_invalid))

        any_endpoint = False
        if method_name in replace_exact:
            mask_low_endpoint = (x == low)
            mask_high_endpoint = (x == high)
            mask_endpoint = (mask_low_endpoint | mask_high_endpoint)
            any_endpoint = (mask_endpoint if mask_endpoint.shape == ()
                            else np.any(mask_endpoint))

        if any_invalid:
            x = np.array(x, dtype=dtype, copy=True)
            x[mask_invalid] = np.nan

        res = np.asarray(f(self, x, *args, **kwargs))

        res_needs_copy = False
        if res.dtype != dtype:
            dtype = np.result_type(dtype, self._dtype)
            res_needs_copy = True

        if res.shape != shape:  # faster to check first
            res = np.broadcast_to(res, self._shape)
            res_needs_copy = res_needs_copy or any_invalid or any_endpoint

        if res_needs_copy:
            res = np.array(res, dtype=dtype, copy=True)

        if any_invalid:
            replace_low, replace_high = (
                replacements.get(method_name, (np.nan, np.nan)))
            res[mask_low] = replace_low
            res[mask_high] = replace_high

        if any_endpoint:
            a, b = self.support()
            if a.shape != shape:
                a = np.array(np.broadcast_to(a, shape), copy=True)
                b = np.array(np.broadcast_to(b, shape), copy=True)

            replace_low_endpoint = (
                b[mask_low_endpoint] if method_name.endswith('ccdf')
                else a[mask_low_endpoint])
            replace_high_endpoint = (
                a[mask_high_endpoint] if method_name.endswith('ccdf')
                else b[mask_high_endpoint])

            res[mask_low_endpoint] = replace_low_endpoint
            res[mask_high_endpoint] = replace_high_endpoint

        return res[()]

    return filtered

def _set_invalid_nan_property(f):
    # maybe implement the cache here?
    def filtered(self, *args, method=None, iv_policy=None, **kwargs):
        if (self.iv_policy or iv_policy) == IV_POLICY.SKIP_ALL:
            return f(self, *args, method=method, **kwargs)

        res = f(self, *args, method=method, **kwargs)
        if res is None:
            # message could be more appropriate
            raise NotImplementedError(self._not_implemented)

        res = np.asarray(res)
        needs_copy = False
        dtype = res.dtype

        if dtype != self._dtype:  # this won't work for logmoments (complex)
            dtype = np.result_type(dtype, self._dtype)
            needs_copy = True

        if res.shape != self._shape:  # faster to check first
            res = np.broadcast_to(res, self._shape)
            needs_copy = needs_copy or self._any_invalid

        if needs_copy:
            res = np.asarray(res, dtype=dtype)

        if self._any_invalid:
            # may be redundant when quadrature is used, but not necessarily
            # when formulas are used.
            res[self._invalid] = np.nan

        return res[()]

    return filtered

def _dispatch(f):

    def wrapped(self, *args, method=None, **kwargs):
        func_name = f.__name__
        method = method or self._method_cache.get(func_name, None)
        if callable(method):
            pass
        elif method is not None:
            method = 'logexp' if method == 'log/exp' else method
            method_name = func_name.replace('dispatch', method)
            method = getattr(self, method_name)
        else:
            method = f(self, *args, method=method, **kwargs)
            self._method_cache[func_name] = method

        return self._call_selected_method(method, *args, **kwargs)

    return wrapped


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
    _parameterizations = []

    def __init__(self, *, tol=_null, iv_policy=None, cache_policy=None,
                 rng=None, **parameters):
        self.tol = tol
        self.iv_policy = iv_policy
        self.cache_policy = (cache_policy if cache_policy is not None
                             else CACHE_POLICY.CACHE)
        self._rng = self._validate_rng(rng, iv_policy)
        self._not_implemented = (
            f"`{self.__class__.__name__}` does not provide an accurate "
            "implementation of the required method. Leave `tol` unspecified "
            "to use the default implementation."
        )
        self._original_parameters = {}

        # Assembling this dictionary increases instantiation time but makes
        # the dispatch methods easier to read and improves method call time.
        # There are a few things we could do to avoid increasing the
        # instantiation time:
        # - make all of these classmethods (so we only get hit at import time)
        # - amortize the creation of this dict by adding the methods
        #   as they are called (e.g. this information could go in the
        #   `_dispatch` wrapper)
        # - compute the name of the method we want to call in @_dispatch, e.g.
        #   func_name.replace('dispatch', method).
        self.update_parameters(**parameters)

    def update_parameters(self, *, iv_policy=None, **kwargs):

        parameters = original_parameters = self._original_parameters.copy()
        parameters.update(**kwargs)
        parameterization = None
        self._invalid = np.asarray(False)
        self._any_invalid = False
        self._shape = tuple()
        self._dtype = np.float64

        if (iv_policy or self.iv_policy) == IV_POLICY.SKIP_ALL:
            pass
        elif not len(self._parameterizations):
            if parameters:
                message = (f"The `{self.__class__.__name__}` distribution "
                           "family does not accept parameters, but parameters "
                           f"`{set(parameters)}` were provided.")
                raise ValueError(message)
        else:
            # This is default behavior, which re-runs all the parameter
            # validation whenever a parameter is modified. For many
            # distributions, the domain of a parameter doesn't depend on
            # other parameters, so parameters could safely be modified
            # withuout validating the other parameters again. To handle
            # these cases more efficiently, we could allow the developer
            # to override this behavior.

            # Currently the user can only update the original parameterization.
            # Even though that parameterization is already known,
            # `_identify_parameterization` is called to produce a nice error
            # message if the user passes other values. To be a little more
            # efficient, we could detect whether the values passed are
            # consistent with the original parameterization rather than finding
            # it from scratch. However, we might want other parameterizations
            # to be accepted, which would require other changes, so I didn't
            # optimize this.

            parameterization = self._identify_parameterization(parameters)
            parameters, shape = self._broadcast(parameters)
            parameters, invalid, any_invalid, dtype = self._validate(
                parameterization, parameters)
            parameters = self._process_parameters(**parameters)

            self._invalid = invalid
            self._any_invalid = any_invalid
            self._shape = shape
            self._dtype = dtype

        self.reset_cache()
        self._parameters = parameters
        self._parameterization = parameterization
        self._original_parameters = original_parameters

    def __getattr__(self, item):
        # This might be needed in __init__ to ensure that `_parameters` exists
        # super().__setattr__('_parameters', dict())

        # This is needed for deepcopy/pickling
        if '_parameters' not in vars(self):
            return super().__getattribute__(item)

        if item in self._parameters:
            return self._parameters[item]

        return super().__getattribute__(item)

    def reset_cache(self):
        # For simplicity, these will still exist even if cache_policy is
        # NO_CACHE; they just won't be populated. This allows caching to be
        # turned on and off easily.
        self._moment_raw_cache = {}
        self._moment_central_cache = {}
        self._moment_standard_cache = {}
        self._support_cache = None
        self._method_cache = {}

    def __repr__(self):
        class_name = self.__class__.__name__
        parameters = list(self._original_parameters)
        info = []
        if parameters:
            parameters.sort()
            info.append(f"{', '.join(parameters)}")
        if self._shape:
            info.append(f"shape={self._shape}")
        if self._dtype != np.float64:
            info.append(f"dtype={self._dtype}")
        return f"{class_name}({', '.join(info)})"

    def _validate_rng(self, rng, iv_policy=None):
        # _validate_rng returns None if rng is None. `default_rng()` takes
        # ~30 µs on my machine.
        if (self.iv_policy or iv_policy) == IV_POLICY.SKIP_ALL:
            return rng

        if rng is not None and not isinstance(rng, np.random.Generator):
            message = ("Argument `rng` passed to the "
                       f"`{self.__class__.__name__}` distribution family is "
                       f"of type `{type(rng)}`, but it must be a NumPy "
                       "`Generator`.")
            raise ValueError(message)
        return rng

    @classmethod
    def _identify_parameterization(cls, parameters):
        # I've come back to this a few times wanting to avoid this explicit
        # loop. I've considered several possibilities, but they've all been a
        # little unusual. For example, we could override `_eq_` so we can
        # use _parameterizations.index() to retrieve the parameterization,
        # or the user could put the parameterizations in a dictionary so we
        # could look them up with a key (e.g. frozenset of parameter names).
        # I haven't been sure enough of these approaches to implement them.
        parameter_names_set = set(parameters)

        for parameterization in cls._parameterizations:
            if parameterization.matches(parameter_names_set):
                break
        else:
            if not parameter_names_set:
                message = (f"The `{cls.__name__}` distribution family "
                           "requires parameters, but none were provided.")
            else:
                parameter_names = cls._get_parameter_str(parameters)
                message = (f"The provided parameters `{parameter_names}` "
                           "do not match a supported parameterization of the "
                           f"`{cls.__name__}` distribution family.")
            raise ValueError(message)

        return parameterization

    @classmethod
    def _broadcast(cls, parameters):
        # broadcast parameters

        # It's much faster to check whether broadcasting is necessary than to
        # broadcast when it's not necessary.
        # We should always make sure that the parameters *are* the same shape
        # and not just broadcastble, though. Users can access parameters as
        # attributes, and I think they should see the arrays as the same shape.
        # More importantly, broadcasting can be important before logical
        # indexing operations, which are needed in infrastructure code when
        # there are invalid paramters, and may be needed in
        # distribution-specific code. We don't want developers to need to
        # broadcast in distribution functions.

        # list(map(np.asarray, parameters.values())) is more compact but less familiar
        parameter_vals = [np.asarray(parameter) for parameter in parameters.values()]
        parameter_shapes = set((parameter.shape for parameter in parameter_vals))
        if len(parameter_shapes) == 1:
            return parameters, parameter_vals[0].shape

        try:
            parameter_vals = np.broadcast_arrays(*parameters.values())
        except ValueError as e:
            parameter_names = cls._get_parameter_str(parameters)
            message = (f"The parameters `{parameter_names}` provided to the "
                       f"`{cls.__name__}` distribution family "
                       "cannot be broadcast to the same shape.")
            raise ValueError(message) from e
        return (dict(zip(parameters.keys(), parameter_vals)),
                parameter_vals[0].shape)

    @classmethod
    def _get_parameter_str(cls, parameters):
        # Sorting for the sake of input validation tests
        parameter_names_list = list(parameters.keys())
        parameter_names_list.sort()
        return f"{{{', '.join(parameter_names_list)}}}"

    @classmethod
    def _validate(cls, parameterization, parameters):
        # Replace invalid parameters with `np.nan` and get NaN pattern
        valid, dtype = parameterization.validation(parameters)
        invalid = ~valid
        any_invalid = invalid if invalid.shape == () else np.any(invalid)
        # If necessary, make the arrays contiguous and replace invalid with NaN
        if any_invalid:
            for parameter_name in parameters:
                parameters[parameter_name] = np.copy(parameters[parameter_name])
                parameters[parameter_name][invalid] = np.nan

        return parameters, invalid, any_invalid, dtype

    def _process_parameters(self, **kwargs):
        return kwargs

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
        super_method = getattr(ContinuousDistribution, method_name, None)
        return method is not super_method

    @property
    def tol(self):
        return self._tol

    @tol.setter
    def tol(self, tol):
        if tol is _null:
            self._tol = tol
            return

        tol = np.asarray(tol)
        if (tol.shape != () or not tol <= 0 or  # catches NaNs
                not np.issubdtype(tol.dtype, np.floating)):
            message = (f"Attribute `tol` of `{self.__class__.__name__}` must "
                       "be a positive float, if specified.")
            raise ValueError(message)
        self._tol = tol[()]

    def plot(self, ax=None, funcs=None, cdf=0.001, ccdf=0.001):
        try:
            import matplotlib  # noqa
        except ModuleNotFoundError as exc:
            message = ("`matplotlib` must be installed to use "
                       f"`{self.__class__.__name__}.plot`.")
            raise ModuleNotFoundError(message) from exc

        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.gca()

        funcs = funcs or ['pdf']
        funcs = [funcs] if type(funcs) == str else funcs

        # Should also offer control over absolute `x` instead of probability
        a, b = self.support()
        a = np.where(np.isinf(a), self.icdf(cdf), a)
        b = np.where(np.isinf(b), self.iccdf(ccdf), b)
        x = np.linspace(a, b, 300)

        for method in funcs:
            f = getattr(self, method)

            def hist_plot(x, *args, **kwargs):
                sample = f(1000)
                # should cut off at user-specified limits
                ax.hist(sample, bins=30, density=True,
                        histtype='step', *args, **kwargs)

            def fun_plot(x, *args, **kwargs):
                y = f(x)
                ax.plot(x, y, *args, **kwargs)

            plot = (hist_plot if method in {'sample', 'qmc_sample'}
                    else fun_plot)

            if len(funcs) > 1:
                # Would be good to have different ylabel scales
                # if there are two methods; create different plots
                # depending on shape of methods nested lists
                plot(x, label=method)
            else:
                # would be good to show parameters as label
                plot(x)

        # should use LaTeX; use symbols
        ax.set_xlabel('x')
        ax.set_ylabel('pdf(x)')
        if len(funcs) > 1:
            ax.legend()
        method_str = (f".{funcs[0]}" if len(funcs) == 1
                      else f" functions {funcs}")
        title = str(self) + method_str
        ax.set_title(title)
        return ax

    def support(self):
        # We manually cache this instead of using `cached_property` for two
        # reasons:
        # - make it easier for users to remember what is a method and what is
        #   an attribute: anything the user can set (e.g. an attribute, policy,
        #   or tolerance) is an attribute; everything that can only return
        #   information is a method.
        # - If this were a `cached_property`, we couldn't update the value
        #   when the distribution parameters change.
        # Caching is important, though, because calls to _support take 1~2 µs
        # even when `a` and `b` are already the same shape.
        if self._support_cache is not None:
            return self._support_cache
        support = self._support(**self._parameters)
        if self.cache_policy != CACHE_POLICY.NO_CACHE:
            self._support_cache = support
        return support

    def _support(self, **kwargs):
        a, b = self._variable.domain.get_numerical_endpoints(kwargs)
        if a.shape != b.shape:
            a, b = np.broadcast_arrays(a, b)
        return a[()], b[()]

    @_set_invalid_nan_property
    def logentropy(self, *, method=None):
        return self._logentropy_dispatch(method=method, **self._parameters)


    def _call_selected_method(self, method, *args, **kwargs):
        try:
            return method(*args, **kwargs)
        except KeyError as e:
            raise NotImplementedError(self._not_implemented) from e


    @_dispatch
    def _logentropy_dispatch(self, method=None, **kwargs):
        if self._overrides('_logentropy_formula'):
            method = self._logentropy_formula
        elif self.tol is _null and self._overrides('_entropy_formula'):
            method = self._logentropy_logexp
        else:
            method = self._logentropy_quadrature
        return method

    def _logentropy_formula(self, **kwargs):
        raise NotImplementedError(self._not_implemented)

    def _logentropy_logexp(self, **kwargs):
        res = np.log(self._entropy_dispatch(**kwargs) + 0j)
        return _log_real_standardize(res)

    def _logentropy_quadrature(self, **kwargs):
        def logintegrand(x, **kwargs):
            logpdf = self._logpdf_dispatch(x, **kwargs)
            return logpdf + np.log(0j+logpdf)
        res = self._quadrature(logintegrand, kwargs=kwargs, log=True)
        return _log_real_standardize(res + np.pi*1j)

    @_set_invalid_nan_property
    def entropy(self, *, method=None):
        return self._entropy_dispatch(method=method, **self._parameters)

    @_dispatch
    def _entropy_dispatch(self, method=None, **kwargs):
        if self._overrides('_entropy_formula'):
            method = self._entropy_formula
        elif self._overrides('_logentropy_formula'):
            method = self._entropy_logexp
        else:
            method = self._entropy_quadrature
        return method

    def _entropy_formula(self, **kwargs):
        raise NotImplementedError(self._not_implemented)

    def _entropy_logexp(self, **kwargs):
        return np.exp(self._logentropy_dispatch(**kwargs))

    def _entropy_quadrature(self, **kwargs):
        def integrand(x, **kwargs):
            pdf = self._pdf_dispatch(x, **kwargs)
            return np.log(pdf)*pdf
        return -self._quadrature(integrand, kwargs=kwargs)

    @_set_invalid_nan_property
    def median(self, *, method=None):
        return self._median_dispatch(method=method, **self._parameters)

    @_dispatch
    def _median_dispatch(self, method=None, **kwargs):
        if self._overrides('_median_formula'):
            method = self._median_formula
        else:
            method = self._median_icdf
        return method

    def _median_formula(self, **kwargs):
        raise NotImplementedError(self._not_implemented)

    def _median_icdf(self, **kwargs):
        return self._icdf_dispatch(0.5, **kwargs)

    @_set_invalid_nan_property
    def mode(self, *, method=None):
        return self._mode_dispatch(method=method, **self._parameters)

    @_dispatch
    def _mode_dispatch(self, method=None, **kwargs):
        # We could add a method that looks for a critical point with
        # differentiation and the root finder
        if self._overrides('_mode_formula'):
            method = self._mode_formula
        else:
            method = self._mode_optimization
        return method

    def _mode_formula(self, **kwargs):
        raise NotImplementedError(self._not_implemented)

    def _mode_optimization(self, **kwargs):
        # Heuristic until we write a proper minimization bracket finder (like
        # bracket_root): if the PDF at the 0.01 and 99.99 percentiles is not
        # less than the PDF at the median, it's either a (rare in SciPy)
        # bimodal distribution (in which case the generic implementation will
        # never be great) or the mode is at one of the endpoints.
        if not np.prod(self._shape):
            return np.empty(self._shape, dtype=self._dtype)
        p_shape = (3,) + (1,)*len(self._shape)
        p = np.asarray([0.0001, 0.5, 0.9999]).reshape(p_shape)
        bracket = self._icdf_dispatch(p, **kwargs)
        res = _chandrupatla_minimize(lambda x: -self._pdf_dispatch(x, **kwargs),
                                     *bracket)
        mode = np.asarray(res.x)
        mode_at_boundary = ~res.success
        mode_at_left = mode_at_boundary & (res.fl <= res.fr)
        mode_at_right = mode_at_boundary & (res.fr < res.fl)
        a, b = self._support(**kwargs)
        mode[mode_at_left] = a[mode_at_left]
        mode[mode_at_right] = b[mode_at_right]
        return mode[()]

    def mean(self, *, method=None, cache_policy=None):
        return self.moment_raw(1, method=method, cache_policy=cache_policy)

    def var(self, *, method=None, cache_policy=None):
        return self.moment_central(2, method=method, cache_policy=cache_policy)

    def std(self, *, method=None, cache_policy=None):
        return np.sqrt(self.var(method=method, cache_policy=cache_policy))

    def skewness(self, *, method=None, cache_policy=None):
        return self.moment_standard(3, method=method, cache_policy=cache_policy)

    def kurtosis(self, *, method=None, cache_policy=None):
        return self.moment_standard(4, method=method, cache_policy=cache_policy)

    @_set_invalid_nan
    def logpdf(self, x, *, method=None):
        return self._logpdf_dispatch(x, method=method, **self._parameters)

    @_dispatch
    def _logpdf_dispatch(self, x, *, method=None, **kwargs):
        if self._overrides('_logpdf_formula'):
            method = self._logpdf_formula
        elif self.tol is _null:  # ensure that developers override _logpdf
            method = self._logpdf_logexp
        return method

    def _logpdf_formula(self, x, **kwargs):
        raise NotImplementedError(self._not_implemented)

    def _logpdf_logexp(self, x, **kwargs):
        return np.log(self._pdf_dispatch(x, **kwargs))

    @_set_invalid_nan
    def pdf(self, x, *, method=None):
        return self._pdf_dispatch(x, method=method, **self._parameters)

    @_dispatch
    def _pdf_dispatch(self, x, *, method=None, **kwargs):
        if self._overrides('_pdf_formula'):
            method = self._pdf_formula
        else:
            method = self._pdf_logexp
        return method

    def _pdf_formula(self, x, **kwargs):
        raise NotImplementedError(self._not_implemented)

    def _pdf_logexp(self, x, **kwargs):
        return np.exp(self._logpdf_dispatch(x, **kwargs))

    @_set_invalid_nan
    def logcdf(self, x, *, method=None):
        return self._logcdf_dispatch(x, method=method, **self._parameters)

    @_dispatch
    def _logcdf_dispatch(self, x, *, method=None, **kwargs):
        if self._overrides('_logcdf_formula'):
            method = self._logcdf_formula
        elif self.tol is _null and self._overrides('_cdf_formula'):
            method = self._logcdf_logexp
        elif self._overrides('_logccdf_formula'):
            method = self._logcdf_complementarity
        else:
            method = self._logcdf_quadrature
        return method

    def _logcdf_formula(self, x, **kwargs):
        raise NotImplementedError(self._not_implemented)

    def _logcdf_logexp(self, x, **kwargs):
        return np.log(self._cdf_dispatch(x, **kwargs))

    def _logcdf_complementarity(self, x, **kwargs):
        return _log1mexp(self._logccdf_dispatch(x, **kwargs))

    def _logcdf_quadrature(self, x, **kwargs):
        a, _ = self._support(**kwargs)
        return self._quadrature(self._logpdf_dispatch, limits=(a, x),
                                kwargs=kwargs, log=True)

    def cdf(self, x, y=None, *, method=None):
        if y is None:
            return self._cdf1(x, method=method)
        else:
            return self._cdf2(x, y, method=method)

    def _cdf2(self, x, y, *, method=None):
        # Draft logic here. Not really correct, and method names are poor.
        if ((self._tol is _null and self._overrides('_cdf')
             and self._overrides('_ccdf') and method is None)
                or method=='formula'):
            return self._cdf2_formula(x, y)
        elif ((self._overrides('_logcdf')
               and self._overrides('_logccdf') and method is None)
              or method=='log/exp'):
            return self._cdf2_logexp(x, y)
        else:
            return self._cdf2_quadrature(x, y)

    def _cdf2_formula(self, x, y):
        # Quick draft. Lots of optimizations possible.
        # - do (custom) input validation once instead of four times
        # - lazy evaluation of ccdf only where it's needed
        # - add logic/options for using logcdf, quadrature
        # - we could stack x and y to reduce number of function calls
        cdf_x = self.cdf(x)
        cdf_y = self.cdf(y)
        ccdf_x = self.ccdf(x)
        ccdf_y = self.ccdf(y)
        i = (cdf_x < 0.5) & (cdf_y < 0.5)
        return np.where(i, cdf_y-cdf_x, ccdf_x-ccdf_y)

    def _cdf2_logexp(self, x, y):
        logcdf_x = self.logcdf(x)
        logcdf_y = self.logcdf(y)
        logccdf_x = self.logccdf(x)
        logccdf_y = self.logccdf(y)
        i = (logcdf_x < -1) & (logcdf_y < -1)
        return np.real(np.exp(np.where(i, _logexpxmexpy(logcdf_y, logcdf_x),
                                       _logexpxmexpy(logccdf_x, logccdf_y))))

    def _cdf2_quadrature(self, x, y):
        x, y = np.broadcast_arrays(x, y)
        shape = np.broadcast_shapes(x.shape, self._shape)
        x = np.broadcast_to(x, self._shape)
        y = np.broadcast_to(y, self._shape)
        return self._quadrature(self._pdf_dispatch, limits=(x, y),
                                kwargs=self._parameters)

    @_set_invalid_nan
    def _cdf1(self, x, *, method):
        return self._cdf_dispatch(x, method=method, **self._parameters)

    @_dispatch
    def _cdf_dispatch(self, x, *, method=None, **kwargs):
        if self._overrides('_cdf_formula'):
            method = self._cdf_formula
        elif self._overrides('_logcdf_formula'):
            method = self._cdf_logexp
        elif self._tol is _null and self._overrides('_ccdf_formula'):
            method = self._cdf_complementarity
        else:
            method = self._cdf_quadrature
        return method

    def _cdf_formula(self, x, **kwargs):
        raise NotImplementedError(self._not_implemented)

    def _cdf_logexp(self, x, **kwargs):
        return np.exp(self._logcdf_dispatch(x, **kwargs))

    def _cdf_complementarity(self, x, **kwargs):
        return 1 - self._ccdf_dispatch(x, **kwargs)

    def _cdf_quadrature(self, x, **kwargs):
        a, _ = self._support(**kwargs)
        return self._quadrature(self._pdf_dispatch, limits=(a, x),
                                kwargs=kwargs)

    @_set_invalid_nan
    def logccdf(self, x, *, method=None):
        return self._logccdf_dispatch(x, method=method, **self._parameters)

    @_dispatch
    def _logccdf_dispatch(self, x, method=None, **kwargs):
        if self._overrides('_logccdf_formula'):
            method = self._logccdf_formula
        elif self.tol is _null and self._overrides('_ccdf_formula'):
            method = self._logccdf_logexp
        elif self._overrides('_logcdf_formula'):
            method = self._logccdf_complementarity
        else:
            method = self._logccdf_quadrature
        return method

    def _logccdf_formula(self):
        return NotImplementedError(self._not_implemented)

    def _logccdf_logexp(self, x, **kwargs):
        return np.log(self._ccdf_dispatch(x, **kwargs))

    def _logccdf_complementarity(self, x, **kwargs):
        return _log1mexp(self._logcdf_dispatch(x, **kwargs))

    def _logccdf_quadrature(self, x, **kwargs):
        _, b = self._support(**kwargs)
        return self._quadrature(self._logpdf_dispatch, limits=(x, b),
                                kwargs=kwargs, log=True)

    @_set_invalid_nan
    def ccdf(self, x, *, method=None):
        return self._ccdf_dispatch(x, method=method, **self._parameters)

    @_dispatch
    def _ccdf_dispatch(self, x, method=None, **kwargs):
        if self._overrides('_ccdf_formula'):
            method = self._ccdf_formula
        elif self._overrides('_logccdf_formula'):
            method = self._ccdf_logexp
        elif self._tol is _null and self._overrides('_cdf_formula'):
            method = self._ccdf_complementarity
        else:
            method = self._ccdf_quadrature
        return method

    def _ccdf_formula(self, x, **kwargs):
        return NotImplementedError(self._not_implemented)

    def _ccdf_logexp(self, x, **kwargs):
        return np.exp(self._logccdf_dispatch(x, **kwargs))

    def _ccdf_complementarity(self, x, **kwargs):
        return 1 - self._cdf_dispatch(x, **kwargs)

    def _ccdf_quadrature(self, x, **kwargs):
        _, b = self._support(**kwargs)
        return self._quadrature(self._pdf_dispatch, limits=(x, b),
                                kwargs=kwargs)

    @_set_invalid_nan
    def ilogcdf(self, x, *, method=None):
        return self._ilogcdf_dispatch(x, method=method, **self._parameters)

    @_dispatch
    def _ilogcdf_dispatch(self, x, method=None, **kwargs):
        if self._overrides('_ilogcdf_formula'):
            method = self._ilogcdf_formula
        elif self._overrides('_ilogccdf_formula'):
            method = self._ilogcdf_complementarity
        else:
            method = self._ilogcdf_inversion
        return method

    def _ilogcdf_formula(self, x, **kwargs):
        raise NotImplementedError(self._not_implemented)

    def _ilogcdf_complementarity(self, x, **kwargs):
        return self._ilogccdf_dispatch(_log1mexp(x), **kwargs)

    def _ilogcdf_inversion(self, x, **kwargs):
        return self._solve_bounded(self._logcdf_dispatch, x, kwargs=kwargs)

    @_set_invalid_nan
    def icdf(self, x, *, method=None):
        return self._icdf_dispatch(x, method=method, **self._parameters)

    @_dispatch
    def _icdf_dispatch(self, x, method=None, **kwargs):
        if self._overrides('_icdf_formula'):
            method = self._icdf_formula
        elif self.tol is _null and self._overrides('_iccdf_formula'):
            method = self._icdf_complementarity
        else:
            method = self._icdf_inversion
        return method

    def _icdf_formula(self, x, **kwargs):
        return NotImplementedError(self._not_implemented)

    def _icdf_complementarity(self, x, **kwargs):
        return self._iccdf_dispatch(1 - x, **kwargs)

    def _icdf_inversion(self, x, **kwargs):
        return self._solve_bounded(self._cdf_dispatch, x, kwargs=kwargs)

    @_set_invalid_nan
    def ilogccdf(self, x, *, method=None):
        return self._ilogccdf_dispatch(x, method=method, **self._parameters)

    @_dispatch
    def _ilogccdf_dispatch(self, x, method=None, **kwargs):
        if self._overrides('_ilogccdf_formula'):
            method = self._ilogccdf_formula
        elif self._overrides('_ilogcdf_formula'):
            method = self._ilogccdf_complementarity
        else:
            method = self._ilogccdf_inversion
        return method

    def _ilogccdf_formula(self, x, **kwargs):
        return NotImplementedError(self._not_implemented)

    def _ilogccdf_complementarity(self, x, **kwargs):
        return self._ilogcdf_dispatch(_log1mexp(x), **kwargs)

    def _ilogccdf_inversion(self, x, **kwargs):
        return self._solve_bounded(self._logccdf_dispatch, x, kwargs=kwargs)

    @_set_invalid_nan
    def iccdf(self, x, *, method=None):
        return self._iccdf_dispatch(x, method=method, **self._parameters)

    @_dispatch
    def _iccdf_dispatch(self, x, method=None, **kwargs):
        if self._overrides('_iccdf_formula'):
            method = self._iccdf_formula
        elif self.tol is _null and self._overrides('_icdf_formula'):
            method = self._iccdf_complementarity
        else:
            method = self._iccdf_inversion
        return method

    def _iccdf_formula(self, x, **kwargs):
        return NotImplementedError(self._not_implemented)

    def _iccdf_complementarity(self, x, **kwargs):
        return self._icdf_dispatch(1 - x, **kwargs)

    def _iccdf_inversion(self, x, **kwargs):
        return self._solve_bounded(self._ccdf_dispatch, x, kwargs=kwargs)

    def sample(self, shape=(), *, method=None, rng=None):
        # needs output validation to ensure that developer returns correct
        # dtype and shape
        sample_shape = (shape,) if not np.iterable(shape) else tuple(shape)
        full_shape = sample_shape + self._shape
        rng = self._validate_rng(rng) or self._rng or np.random.default_rng()
        return self._sample_dispatch(sample_shape, full_shape, method=method,
                                     rng=rng, **self._parameters)

    @_dispatch
    def _sample_dispatch(self, sample_shape, full_shape, *, method, rng, **kwargs):
        # make sure that tests catch if sample is 0d array
        if self._overrides('_sample_formula'):
            method = self._sample_formula
        else:
            method = self._sample_inverse_transform
        return method

    def _sample_formula(self, sample_shape, full_shape, *, rng, **kwargs):
        raise NotImplementedError(self._not_implemented)

    def _sample_inverse_transform(self, sample_shape, full_shape, *, rng, **kwargs):
        uniform = rng.uniform(size=full_shape)
        return self._icdf_dispatch(uniform, **kwargs)

    def qmc_sample(self, shape=(), *, method=None, qrng=Halton, rng=None):
        # Any shape is allowed, but observations along the 0th dimension are
        # not independent.
        sample_shape = (shape,) if not np.iterable(shape) else tuple(shape)
        full_shape = sample_shape + self._shape
        d = int(np.prod(full_shape[1:]))
        length = full_shape[0] if full_shape else 1
        qrng = qrng(d=d, seed=rng)
        return self._qmc_sample_dispatch(
            length, full_shape, method=method, qrng=qrng,
            **self._parameters)

    @_dispatch
    def _qmc_sample_dispatch(self, length, full_shape, *, method, qrng, **kwargs):
        # make sure that tests catch if sample is 0d array
        if self._overrides('_qmc_sample_formula'):
            method = self._qmc_sample_formula
        else:
            method = self._qmc_sample_inverse_transform
        return method

    def _qmc_sample_formula(self, length, full_shape, *, qrng, **kwargs):
        raise NotImplementedError(self._not_implemented)

    def _qmc_sample_inverse_transform(self, length, full_shape, *, qrng, **kwargs):
        uniform = qrng.random(length)
        uniform = np.reshape(uniform, full_shape)
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

    def _validate_order(self, order, f_name, iv_policy=None):
        if (self.iv_policy or iv_policy) == IV_POLICY.SKIP_ALL:
            return order

        order = np.asarray(order, dtype=self._dtype)[()]
        message = (f"Argument `order` of `{self.__class__.__name__}.{f_name}` "
                   "must be a finite, positive integer.")
        try:
            order_int = round(order.item())
            # If this fails for any reason (e.g. it's an array, it's infinite)
            # it's not a valid `order`.
        except Exception as e:
            raise ValueError(message) from e

        if order_int <0 or order_int != order:
            raise ValueError(message)

        return order

    @cached_property
    def _moment_methods(self):
        return {'cache', 'formula', 'transform',
                'normalize', 'general', 'quadrature'}

    @_set_invalid_nan_property
    def moment_raw(self, order=1, *, method=None, cache_policy=None):
        order = self._validate_order(order, "moment_raw")
        methods = self._moment_methods if method is None else {method}
        cache_policy = self.cache_policy if cache_policy is None else cache_policy
        return self._moment_raw_dispatch(
            order, methods=methods, cache_policy=cache_policy, **self._parameters)

    def _moment_raw_dispatch(self, order, *, methods, cache_policy=None, **kwargs):
        # How to indicate to the user if the requested methods could not be used?
        # Rather than returning None when a moment is not available, raise?
        moment = None

        if 'cache' in methods:
            moment = self._moment_raw_cache.get(order, None)

        if moment is None and 'formula' in methods:
            moment = self._moment_raw(order, **kwargs)

        if moment is None and 'transform' in methods and order > 1:
            moment = self._moment_raw_transform(order, **kwargs)

        if moment is None and 'general' in methods:
            moment = self._moment_raw_general(order, **kwargs)

        if moment is None and 'quadrature' in methods:
            moment = self._moment_integrate_pdf(order, center=0, **kwargs)

        if moment is not None and cache_policy != CACHE_POLICY.NO_CACHE:
            self._moment_raw_cache[order] = moment

        return moment

    def _moment_raw(self, order, **kwargs):
        return None

    def _moment_raw_transform(self, order, **kwargs):
        central_moments = []
        for i in range(int(order) + 1):
            methods = {'cache', 'formula', 'normalize', 'general'}
            moment_i = self._moment_central_dispatch(order=i,
                                                     methods=methods, **kwargs)
            if moment_i is None:
                return None
            central_moments.append(moment_i)

        # Doesn't make sense to get the mean by "transform", since that's
        # how we got here. Questionable whether 'quadrature' should be here.
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
    def moment_central(self, order=1, *, method=None, cache_policy=None):
        order = self._validate_order(order, "moment_central")
        methods = self._moment_methods if method is None else {method}
        cache_policy = self.cache_policy if cache_policy is None else cache_policy
        return self._moment_central_dispatch(
            order, methods=methods, cache_policy=cache_policy, **self._parameters)

    def _moment_central_dispatch(self, order, *, methods, cache_policy=None, **kwargs):
        moment = None

        if 'cache' in methods:
            moment = self._moment_central_cache.get(order, None)

        if moment is None and 'formula' in methods:
            moment = self._moment_central(order, **kwargs)

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

        if moment is not None and cache_policy != CACHE_POLICY.NO_CACHE:
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
    def moment_standard(self, order=1, *, method=None, cache_policy=None):
        order = self._validate_order(order, "moment_standard")
        methods = self._moment_methods if method is None else {method}
        cache_policy = self.cache_policy if cache_policy is None else cache_policy
        return self._moment_standard_dispatch(
            order, methods=methods, cache_policy=cache_policy, **self._parameters)

    def _moment_standard_dispatch(self, order, *, methods, cache_policy=None, **kwargs):
        moment = None

        if 'cache' in methods:
            moment = self._moment_standard_cache.get(order, None)

        if moment is None and 'formula' in methods:
            moment = self._moment_standard(order, **kwargs)

        if moment is None and 'normalize' in methods:
            moment = self._moment_standard_transform(order, False, **kwargs)

        if moment is None and 'general' in methods:
            moment = self._moment_standard_general(order, **kwargs)

        if moment is None and 'normalize' in methods:
            moment = self._moment_standard_transform(order, True, **kwargs)

        if moment is not None and cache_policy != CACHE_POLICY.NO_CACHE:
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
        a, b = np.broadcast_arrays(a, b)
        if not a.size:
            # maybe need to figure out result type from a, b
            return np.empty(a.shape, dtype=self._dtype)
        args = [] if args is None else args
        kwargs = {} if kwargs is None else kwargs
        f, args = kwargs2args(integrand, args=args, kwargs=kwargs)
        args = np.broadcast_arrays(*args)
        res = _tanhsinh(f, a, b, args=args, log=log)
        return res.integral

    def _solve_bounded(self, f, p, *, bounds=None, kwargs=None):
        # should modify _bracket_root and _chandrupatla so we don't need all this
        min, max = self._support(**kwargs) if bounds is None else bounds
        kwargs = {} if kwargs is None else kwargs

        p, min, max = np.broadcast_arrays(p, min, max)
        if not p.size:
            # might need to figure out result type based on p
            return np.empty(p.shape, dtype=self._dtype)

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

    # At first glance, it would seem ideal for `fit` were a classmethod,
    # called like `LogUniform.fit(sample=sample)`.
    # I tried this. I insisted on it for a while. Consider the following:
    # * if `fit` is a classmethod, it cannot call instance methods. If we
    #   want to support MLE, MPS, MoM, MoLM, then we end up with most of the
    #   private methods needing to be classmethods, too. There are two issues
    #   with this.
    #   - Although they are nearly independent of state and get called with
    #     the parameters as keyword arguments, they do depend on some instance
    #     attributes like `tol` and the cache policy. These sorts of things
    #     can be passed in, but then there are other cases like the overridden
    #     methods of `OrderStatisticDistribution` and
    #     `ShiftedScaledDistribution`, which naturally want to have access to
    #     `self._dist`, the distribution they modify. If we want to pass that
    #     in, we would have to modify the public methods to do so. Do we
    #     really want to modify all that just for `fit`, `llf`? And do we
    #     really want to see the @classmethod decorator on pretty much every
    #     method we write, including the overrides for specified
    #     distributions?
    #   - Even more importantly: we will want to `fit` and get the LLF for
    #     distributions that aren't simple instances of named classes, but they
    #     are transformed from standard distributions. Does it make sense to
    #     call `ShiftedScaledDistribution.fit...` and pass in the class of the
    #     distribution that is shifted/scaled as an addtitional argument?
    #     In such cases, I think it is more intuitive for the user to
    #     generate such a transformed distribution `dist` and call `dist.fit`.
    #     Fortunately, this essentially *requires* the user to provide a guess
    #     of the parameters, which is actually a good thing in the general
    #     case (no analytical formula).
    # There are ways of getting around these issues, and I've explored them.
    # But the fact is that this infrastructure is object-oriented. There are a
    # *lot* of advantages to this, so I would rather not fight it for the few
    # cases where we might prefer a slightly different invocation as a matter
    # of principle.

    def llf(self, parameters={}, *, sample, axis=-1):
        self.update_parameters(**parameters)
        return np.sum(self.logpdf(sample), axis=axis)

    def dllf(self, parameters={}, *, sample, var):
        self.update_parameters(**parameters)

        def f(x):
            update = {}
            update[var] = x
            self.update_parameters(**update)
            res = self.llf(sample=sample[:, np.newaxis], axis=0)
            return np.reshape(res, x.shape)

        return _differentiate(f, self._parameters[var]).df

    def fit(self, sample):
        # very basic `fit` method that only works for distributions with
        # unbounded parameter and argument domains.
        names = list(self._original_parameters.keys())
        x0 = list(self._original_parameters.values())
        def objective(x):
            self.update_parameters(**dict(zip(names, x)))
            return -self.llf(sample=sample)

        res = optimize.minimize(objective, x0)
        return res

# Rough sketch of how we might shift/scale distributions. The purpose of
# making it a separate class is just for
# a) simplicity of the ContinuousDistribution class and
# b) avoiding the requirement that every distribution accept loc/scale.
# The simplicity of ContinuousDistribution is important, because there are
# several other distribution transformations to be supported; e.g., truncation,
# wrapping, folding, and doubling. We wouldn't want to cram all of this
# into the `ContinuousDistribution` class. Also, the order of the composition
# matters (e.g. truncate then shift/scale or vice versa). It's easier to
# accommodate different orders if the transformation is built up from
# components rather than all built into `ContinuousDistribution`.

class ShiftedScaledDistribution(ContinuousDistribution):
    _loc_domain = _RealDomain(endpoints=(-oo, oo), inclusive=(False, False))
    _loc_param = _RealParameter('loc', symbol='µ', domain=_loc_domain, typical=(1, 2))

    _scale_domain = _RealDomain(endpoints=(-oo, oo), inclusive=(False, False))
    _scale_param = _RealParameter('scale', symbol='σ', domain=_scale_domain, typical=(0.1, 10))

    _parameterizations = [_Parameterization(_loc_param, _scale_param),
                          _Parameterization(_loc_param),
                          _Parameterization(_scale_param)]

    def __init__(self, dist, *args, **kwargs):
        self._dist = dist
        if dist._parameterization:
            # Add standard distribution parameters to our parameterization
            dist_parameters = dist._parameterization.parameters
            for parameterization in self._parameterizations:
                parameterization.parameters.update(dist_parameters)
        super().__init__(*args, **kwargs)

    def _process_parameters(self, loc=None, scale=None, **kwargs):
        loc = loc if loc is not None else np.zeros_like(scale)
        scale = scale if scale is not None else np.ones_like(loc)
        sign = scale > 0
        parameters = self._dist._process_parameters(**kwargs)
        parameters.update(dict(loc=loc, scale=scale, sign=sign))
        return parameters

    def update_parameters(self, *, iv_policy=None, **kwargs):
        # maybe broadcast everything before processing?
        parameters = {}
        # There may be some issues with _original_parameters
        # We only want to update with _dist._original_parameters during
        # initialization. Afterward that, we want to start with
        # self._original_parameters.
        parameters.update(self._dist._original_parameters)
        parameters.update(kwargs)
        super().update_parameters(iv_policy=iv_policy, **parameters)

    def __repr__(self):
        s = super().__repr__()
        return s.replace('ShiftedScaledDistribution',
                         self._dist.__class__.__name__)

    def _transform(self, x, loc, scale, **kwargs):
        return (x - loc)/scale

    def _itransform(self, x, loc, scale, **kwargs):
        return x * scale + loc

    # Because we're supporting negative scale, these methods take twice as
    # long as they would otherwise. Perhaps support for negative scale
    # should be an option.
    def __getattribute__(self, item):
        if item in {'_logcdf_dispatch', '_cdf_dispatch', '_logccdf_dispatch', '_ccdf_dispatch'}:
            f = getattr(self._dist, item)
            citem = {'_logcdf_dispatch': '_logccdf_dispatch', '_cdf_dispatch': '_ccdf_dispatch',
                     '_logccdf_dispatch': '_logcdf_dispatch', '_ccdf_dispatch': '_cdf_dispatch'}
            cf = getattr(self._dist, citem[item])

            def wrapped(x, *args, loc, scale, sign, **kwargs):
                fx = f(self._transform(x, loc, scale), *args, **kwargs)
                cfx = cf(self._transform(x, loc, scale), *args, **kwargs)
                return np.where(sign, fx, cfx)[()]

            return wrapped

        elif item in {'_ilogcdf_dispatch', '_icdf_dispatch', '_ilogccdf_dispatch', '_iccdf_dispatch'}:
            f = getattr(self._dist, item)
            citem = {'_ilogcdf_dispatch': '_ilogccdf_dispatch', '_icdf_dispatch': '_iccdf_dispatch',
                     '_ilogccdf_dispatch': '_ilogcdf_dispatch', '_iccdf_dispatch': '_icdf_dispatch'}
            cf = getattr(self._dist, citem[item])

            def wrapped(x, *args, loc, scale, sign, **kwargs):
                fx = self._itransform(f(x, *args, **kwargs), loc, scale)
                cfx = self._itransform(cf(x, *args, **kwargs), loc, scale)
                return np.where(sign, fx, cfx)[()]

            return wrapped

        return super().__getattribute__(item)

    def _support(self, loc, scale, sign, **kwargs):
        a, b = self._dist._support(**kwargs)
        a, b = self._itransform(a, loc, scale), self._itransform(b, loc, scale)
        return np.where(sign, a, b)[()], np.where(sign, b, a)[()]

    def _entropy_dispatch(self, *args, loc, scale, sign, **kwargs):
        return self._dist._entropy_dispatch(*args, **kwargs) + np.log(abs(scale))

    def _logentropy_dispatch(self, *args, loc, scale, sign, **kwargs):
        lH0 = self._dist._logentropy_dispatch(*args, **kwargs)
        lls = np.log(np.log(abs(scale)))
        return special.logsumexp(np.broadcast_arrays(lH0, lls), axis=0)

    def _median_dispatch(self, *, method, loc, scale, **kwargs):
        raw = self._dist._median_dispatch(method=method, **kwargs)
        return self._itransform(raw, loc, scale)

    def _mode_dispatch(self, *, method, loc, scale, **kwargs):
        raw = self._dist._mode_dispatch(method=method, **kwargs)
        return self._itransform(raw, loc, scale)

    def _logpdf_dispatch(self, x, *args, loc, scale, sign, **kwargs):
        return (self._dist._logpdf_dispatch(self._transform(x, loc, scale), *args, **kwargs)
                - np.log(abs(scale)))

    def _pdf_dispatch(self, x, *args, loc, scale, sign, **kwargs):
        return (self._dist._pdf_dispatch(self._transform(x, loc, scale), *args, **kwargs)
                / abs(scale))

    def _moment_standard_dispatch(self, order, *, loc, scale, methods, cache_policy=None, **kwargs):
        return (self._dist._moment_standard_dispatch(
            order, methods=methods, cache_policy=cache_policy, **kwargs)
                * np.sign(scale)**order)

    def _moment_central_dispatch(self, order, *, loc, scale, methods, cache_policy=None, **kwargs):
        return (self._dist._moment_central_dispatch(
            order, methods=methods, cache_policy=cache_policy, **kwargs)
                * scale**order)

    def _moment_raw_dispatch(self, order, *, loc, scale, methods, cache_policy=None, **kwargs):
        raw_moments = []
        for i in range(int(order) + 1):
            raw = self._dist._moment_raw_dispatch(
                i, methods=methods, cache_policy=cache_policy, **kwargs)
            moment_i = raw * scale**i
            raw_moments.append(moment_i)

        return self._moment_transform_center(
            order, raw_moments, loc, 0)

    def _sample_dispatch(self, sample_shape, full_shape, *, method, rng, **kwargs):
        rvs = self._dist._sample_dispatch(
            sample_shape, full_shape, method=method, rng=rng, **kwargs)
        return rvs*self.scale + self.loc

    def _qmc_sample_dispatch(self, length, full_shape, *, method, qrng, **kwargs):
        rvs = self._dist._qmc_sample_dispatch(
            length, full_shape, method=method, qrng=qrng, **kwargs)
        return rvs*self.scale + self.loc

    # Add these methods to ContinuousDistribution so they can return a
    # ShiftedScaledDistribution
    def __add__(self, loc):
        self.update_parameters(loc=self.loc + loc)
        return self

    def __sub__(self, loc):
        self.update_parameters(loc=self.loc - loc)
        return self

    def __mul__(self, scale):
        self.update_parameters(loc=self.loc * scale,
                               scale=self.scale * scale)
        return self

    def __truediv__(self, scale):
        self.update_parameters(loc=self.loc / scale,
                               scale=self.scale / scale)
        return self

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__add__(other)

    def __rtruediv__(self, other):
        return self.__add__(other)
