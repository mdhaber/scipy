import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from hypothesis import strategies, given, reproduce_failure  # noqa
import hypothesis.extra.numpy as npst

from scipy.stats._fit import _kolmogorov_smirnov
from scipy.stats._ksstats import kolmogn

from scipy.stats._distribution_infrastructure import (
    oo, _Domain, _RealDomain, _RealParameter)
from scipy.stats._new_distributions import LogUniform, Normal

class Test_RealDomain:
    rng = np.random.default_rng(349849812549824)

    @pytest.mark.parametrize('x', [rng.uniform(10, 10, size=(2, 3, 4)),
                                   -np.inf, np.pi])
    def test_contains_simple(self, x):
        # Test `contains` when endpoints are defined by constants
        a, b = -np.inf, np.pi
        domain = _RealDomain(endpoints=(a, b), inclusive=(False, True))
        assert_equal(domain.contains(x), (a < x) & (x <= b))

    @given(shapes=npst.mutually_broadcastable_shapes(num_shapes=3, min_side=0),
           inclusive_a=strategies.booleans(),
           inclusive_b=strategies.booleans(),
           data=strategies.data())
    def test_contains(self, shapes, inclusive_a, inclusive_b, data):
        # Test `contains` when endpoints are defined by parameters
        input_shapes, result_shape = shapes
        shape_a, shape_b, shape_x = input_shapes

        # Without defining min and max values, I spent forever trying to set
        # up a valid test without overflows or similar just drawing arrays.
        a_elements = dict(allow_nan=False, allow_infinity=False,
                          min_value=-1e3, max_value=1)
        b_elements = dict(allow_nan=False, allow_infinity=False,
                          min_value=2, max_value=1e3)
        a = data.draw(npst.arrays(npst.floating_dtypes(),
                                  shape_a, elements=a_elements))
        b = data.draw(npst.arrays(npst.floating_dtypes(),
                                  shape_b, elements=b_elements))
        # ensure some points are to the left, some to the right, and some
        # are exactly on the boundary
        d = b - a
        x = np.concatenate([np.linspace(a-d, a, 10),
                            np.linspace(a, b, 10),
                            np.linspace(b, b+d, 10)])
        # Domain is defined by two parameters, 'a' and 'b'
        domain = _RealDomain(endpoints=('a', 'b'),
                             inclusive=(inclusive_a, inclusive_b))
        domain.define_parameters(_RealParameter('a', domain=_RealDomain()),
                                 _RealParameter('b', domain=_RealDomain()))
        # Check that domain and string evaluation give the same result
        res = domain.contains(x, dict(a=a, b=b))

        # Apparently, `np.float16([2]) < np.float32(2.0009766)` is False
        # but `np.float16([2]) < np.float32([2.0009766])` is True
        dtype = np.result_type(a.dtype, b.dtype, x.dtype)
        a, b, x = a.astype(dtype), b.astype(dtype), x.astype(dtype)
        left_comparison = '<=' if inclusive_a else '<'
        right_comparison = '<=' if inclusive_b else '<'
        ref = eval(f'(a {left_comparison} x) & (x {right_comparison} b)')
        assert_equal(res, ref)

    @pytest.mark.parametrize('case', [
        (-np.inf, np.pi, False, True, "(-∞, π]"),
        ('a', 5, True, False, "[a, 5)")
    ])
    def test_str(self, case):
        domain = _RealDomain(endpoints=case[:2], inclusive=case[2:4])
        assert str(domain) == case[4]

    @given(a=strategies.one_of(strategies.decimals(allow_nan=False),
                               strategies.characters(whitelist_categories="L"),
                               strategies.sampled_from(list(_Domain.symbols))),
           b=strategies.one_of(strategies.decimals(allow_nan=False),
                               strategies.characters(whitelist_categories="L"),
                               strategies.sampled_from(list(_Domain.symbols))),
           inclusive_a=strategies.booleans(),
           inclusive_b=strategies.booleans(),
           )
    def test_str2(self, a, b, inclusive_a, inclusive_b):
        # I wrote this independently from the implementation of __str__, but
        # I imagine it looks pretty similar to __str__.
        a = _Domain.symbols.get(a, a)
        b = _Domain.symbols.get(b, b)
        left_bracket = '[' if inclusive_a else '('
        right_bracket = ']' if inclusive_b else ')'
        domain = _RealDomain(endpoints=(a, b),
                             inclusive=(inclusive_a, inclusive_b))
        ref = f"{left_bracket}{a}, {b}{right_bracket}"
        assert str(domain) == ref


def draw_distribution_from_family(family, data, rng, proportions):
    # If the distribution has parameters, choose a parameterization and
    # draw broadcastable shapes for the parameter arrays.
    n_parameters = family._num_parameters()
    if n_parameters > 0:
        shapes, result_shape = data.draw(
            npst.mutually_broadcastable_shapes(num_shapes=n_parameters,
                                               min_side=0))
        dist = family._draw(shapes, rng=rng, proportions=proportions)
    else:
        dist = family._draw(rng=rng)
        result_shape = tuple()

    # Draw a broadcastable shape for the arguments, and draw values for the
    # arguments.
    x_shape = data.draw(npst.broadcastable_shapes(result_shape, min_side=0))
    x = dist._variable.draw(x_shape, parameter_values=dist._parameters,
                            proportions=proportions)
    x_result_shape = np.broadcast_shapes(x_shape, result_shape)
    p_domain = _RealDomain((0, 1), (True, True))
    p = p_domain.draw(x_shape, proportions=proportions)
    logp = np.log(p)

    return dist, x, p, logp, result_shape, x_result_shape


class TestDistributions:

    @pytest.mark.filterwarnings("ignore")
    @pytest.mark.parametrize('family', (Normal, LogUniform))
    @given(data=strategies.data(), seed=strategies.integers(min_value=0))
    def test_basic(self, family, data, seed):
        # strengthen this test by letting min_side=0 for both broadcasted shapes
        # check for scalar output if all inputs are scalar
        # inject bad parameters and x-values, check NaN pattern
        rng = np.random.default_rng(seed)

        # relative proportions of valid, endpoint, out of bounds, and NaN params
        proportions = (1, 1, 1, 1)
        tmp = draw_distribution_from_family(family, data, rng, proportions)
        dist, x, p, logp, result_shape, x_result_shape = tmp
        sample_shape = data.draw(npst.array_shapes(min_dims=0, min_side=0))

        methods = {'log/exp', 'quadrature'}
        check_dist_func(dist, 'entropy', None, result_shape, methods)
        check_dist_func(dist, 'logentropy', None, result_shape, methods)

        methods = {'icdf'}
        check_dist_func(dist, 'median', None, result_shape, methods)

        methods = {'optimization'}
        check_dist_func(dist, 'mode', None, result_shape, methods)

        methods = {'cache'}  #  weak test right now
        check_dist_func(dist, 'mean', None, result_shape, methods)
        check_dist_func(dist, 'var', None, result_shape, methods)
        check_dist_func(dist, 'skewness', None, result_shape, methods)
        check_dist_func(dist, 'kurtosis', None, result_shape, methods)
        assert_allclose(dist.std()**2, dist.var())

        check_moment_funcs(dist, result_shape)
        check_sample_shape_NaNs(dist, sample_shape, result_shape)

        methods = {'log/exp'}
        check_dist_func(dist, 'pdf', x, x_result_shape, methods)
        check_dist_func(dist, 'logpdf', x, x_result_shape, methods)

        methods = {'log/exp', 'complementarity', 'quadrature'}
        check_dist_func(dist, 'logcdf', x, x_result_shape, methods)
        check_dist_func(dist, 'cdf', x, x_result_shape, methods)
        check_dist_func(dist, 'logccdf', x, x_result_shape, methods)
        check_dist_func(dist, 'ccdf', x, x_result_shape, methods)

        methods = {'complementarity', 'inversion'}
        check_dist_func(dist, 'ilogcdf', logp, x_result_shape, methods)
        check_dist_func(dist, 'icdf', p, x_result_shape, methods)
        check_dist_func(dist, 'ilogccdf', logp, x_result_shape, methods)
        check_dist_func(dist, 'iccdf', p, x_result_shape, methods)


def check_sample_shape_NaNs(dist, sample_shape, result_shape):
    methods = {'inverse_transform'}
    if dist._overrides('_sample'):
        methods.add('formula')

    for method in methods:
        res = dist.sample(sample_shape, method=method)
        valid_parameters = np.broadcast_to(get_valid_parameters(dist),
                                           res.shape)
        assert_equal(res.shape, sample_shape + result_shape)
        assert np.all(np.isfinite(res[valid_parameters]))
        assert_equal(res[~valid_parameters], np.nan)

        sample1 = dist.sample(sample_shape, method=method,
                              rng=np.random.default_rng(42))
        sample2 = dist.sample(sample_shape, method=method,
                              rng=np.random.default_rng(42))
        assert not np.any(np.equal(res, sample1))
        assert_equal(sample1, sample2)


def check_dist_func(dist, fname, arg, result_shape, methods):
    # Check that all computation methods of all distribution functions agree
    # with one another, effectively testing the correctness of the generic
    # computation methods and confirming the consistency of specific
    # distributions with their pdf/logpdf.

    args = tuple() if arg is None else (arg,)
    methods = methods.copy()

    if "cache" in methods:
        # If "cache" is specified before the value has been evaluated, it
        # raises an error. After the value is evaluated, it will succeed.
        with pytest.raises(NotImplementedError):
            getattr(dist, fname)(*args, method="cache")

    ref = getattr(dist, fname)(*args)
    check_nans_and_edges(dist, fname, arg, ref)

    # Remove this after fixing `draw`
    tol_override = {'atol': 1e-15}
    # Mean can be 0, which makes logmean -oo.
    if fname in {'logmean', 'mean', 'logskewness', 'skewness'}:
        tol_override = {'atol': 1e-15}
    elif fname in {'mode'}:
        # can only expect about half of machine precision for optimization
        # because math
        tol_override = {'atol': 1e-8}

    if dist._overrides(f'_{fname}'):
        methods.add('formula')

    np.testing.assert_equal(ref.shape, result_shape)
    # Until we convert to array API, let's do the familiar thing:
    # 0d things are scalars, not arrays
    if result_shape == tuple():
        assert np.isscalar(ref)

    for method in methods:
        res = getattr(dist, fname)(*args, method=method)
        if 'log' in fname:
            np.testing.assert_allclose(np.exp(res), np.exp(ref),
                                       **tol_override)
        else:
            np.testing.assert_allclose(res, ref, **tol_override)

        np.testing.assert_equal(res.shape, result_shape)
        if result_shape == tuple():
            assert np.isscalar(res)

def check_nans_and_edges(dist, fname, arg, res):
    inverses = {'icdf', 'ilogccdf', 'iccdf'}

    valid_parameters = get_valid_parameters(dist)
    if fname in {'icdf', 'iccdf'}:
        arg_domain = _RealDomain(endpoints=(0, 1), inclusive=(True, True))
    elif fname in {'ilogcdf', 'ilogccdf'}:
        arg_domain = _RealDomain(endpoints=(-oo, 0), inclusive=(True, True))
    else:
        arg_domain = dist._variable.domain

    classified_args = classify_arg(dist, arg, arg_domain)
    valid_parameters, *classified_args = np.broadcast_arrays(valid_parameters,
                                                             *classified_args)
    valid_arg, endpoint_arg, outside_arg, nan_arg = classified_args
    all_valid = valid_arg & valid_parameters

    # Check NaN pattern and edge cases
    assert_equal(res[~valid_parameters], np.nan)
    assert_equal(res[nan_arg], np.nan)

    a, b = dist.support
    a = np.broadcast_to(a, res.shape)
    b = np.broadcast_to(b, res.shape)

    # Writing this independently of how the are set in the distribution
    # infrastructure. That is very compact; this is very verbose.
    if fname in {'logpdf'}:
        assert_equal(res[outside_arg == -1], -np.inf)
        assert_equal(res[outside_arg == 1], -np.inf)
        assert_equal(res[(endpoint_arg == -1) & ~valid_arg], -np.inf)
        assert_equal(res[(endpoint_arg == 1) & ~valid_arg], -np.inf)
    elif fname in {'pdf'}:
        assert_equal(res[outside_arg == -1], 0)
        assert_equal(res[outside_arg == 1], 0)
        assert_equal(res[(endpoint_arg == -1) & ~valid_arg], 0)
        assert_equal(res[(endpoint_arg == 1) & ~valid_arg], 0)
    elif fname in {'logcdf'}:
        assert_equal(res[outside_arg == -1], -oo)
        assert_equal(res[outside_arg == 1], 0)
        assert_equal(res[endpoint_arg == -1], -oo)
        assert_equal(res[endpoint_arg == 1], 0)
    elif fname in {'cdf'}:
        assert_equal(res[outside_arg == -1], 0)
        assert_equal(res[outside_arg == 1], 1)
        assert_equal(res[endpoint_arg == -1], 0)
        assert_equal(res[endpoint_arg == 1], 1)
    elif fname in {'logccdf'}:
        assert_equal(res[outside_arg == -1], 0)
        assert_equal(res[outside_arg == 1], -oo)
        assert_equal(res[endpoint_arg == -1], 0)
        assert_equal(res[endpoint_arg == 1], -oo)
    elif fname in {'ccdf'}:
        assert_equal(res[outside_arg == -1], 1)
        assert_equal(res[outside_arg == 1], 0)
        assert_equal(res[endpoint_arg == -1], 1)
        assert_equal(res[endpoint_arg == 1], 0)
    elif fname in {'ilogcdf', 'icdf'}:
        assert_equal(res[outside_arg == -1], np.nan)
        assert_equal(res[outside_arg == 1], np.nan)
        assert_equal(res[endpoint_arg == -1], a[endpoint_arg == -1])
        assert_equal(res[endpoint_arg == 1], b[endpoint_arg == 1])
    elif fname in {'ilogccdf', 'iccdf'}:
        assert_equal(res[outside_arg == -1], np.nan)
        assert_equal(res[outside_arg == 1], np.nan)
        assert_equal(res[endpoint_arg == -1], b[endpoint_arg == -1])
        assert_equal(res[endpoint_arg == 1], a[endpoint_arg == 1])

    if fname not in {'logmean', 'mean', 'logskewness', 'skewness'}:
        assert np.isfinite(res[all_valid & (endpoint_arg == 0)]).all()

def check_moment_funcs(dist, result_shape):
    # Check that all computation methods of all distribution functions agree
    # with one another, effectively testing the correctness of the generic
    # computation methods and confirming the consistency of specific
    # distributions with their pdf/logpdf.

    atol = 1e-10  # make this tighter (e.g. 1e-13) after fixing `draw`

    def check(moment, order, method=None, ref=None, success=True):
        if success:
            res = moment(order, method=method)
            assert_allclose(res, ref, atol=atol)
            assert res.shape == ref.shape
        else:
            with pytest.raises(NotImplementedError):
                moment(order, method=method)

    formula_raw = dist._overrides('_moment_raw')
    formula_central = dist._overrides('_moment_central')
    formula_standard = dist._overrides('_moment_standard')

    dist._moment_raw_cache = {}
    dist._moment_central_cache = {}
    dist._moment_standard_cache = {}

    ### Check Raw Moments ###
    for i in range(6):
        check(dist.moment_raw, i, 'cache', success=False)  # not cached yet
        ref = dist.moment_raw(i, method='quadrature')
        check_nans_and_edges(dist, 'moment_raw', None, ref)
        assert ref.shape == result_shape
        check(dist.moment_raw, i, 'cache', ref, success=True)  # cached now
        check(dist.moment_raw, i, 'formula', ref, success=formula_raw)
        check(dist.moment_raw, i, 'general', ref, i == 0)

    # Clearing caches to better check their behavior
    dist._moment_raw_cache = {}
    dist._moment_central_cache = {}
    dist._moment_standard_cache = {}

    # If we have central or standard moment formulas, or if there are
    # values in their cache, we can use method='transform'
    dist.moment_central(0)  # build up the cache
    dist.moment_central(1)
    for i in range(2, 6):
        ref = dist.moment_raw(i, method='quadrature')
        check(dist.moment_raw, i, 'transform', ref,
              success=formula_central or formula_standard)
        dist.moment_central(i)  # build up the cache
        check(dist.moment_raw, i, 'transform', ref)

    dist._moment_raw_cache = {}
    dist._moment_central_cache = {}
    dist._moment_standard_cache = {}

    ### Check Central Moments ###

    for i in range(6):
        check(dist.moment_central, i, 'cache', success=False)
        ref = dist.moment_central(i, method='quadrature')
        check_nans_and_edges(dist, 'moment_central', None, ref)
        assert ref.shape == result_shape
        check(dist.moment_central, i, 'cache', ref, success=True)
        check(dist.moment_central, i, 'formula', ref, success=formula_central)
        check(dist.moment_central, i, 'general', ref, success=i <= 1)
        check(dist.moment_central, i, 'transform', ref, success=formula_raw)
        if not formula_raw:
            dist.moment_raw(i)
            check(dist.moment_central, i, 'transform', ref)

    dist._moment_raw_cache = {}
    dist._moment_central_cache = {}
    dist._moment_standard_cache = {}

    # If we have standard moment formulas, or if there are
    # values in their cache, we can use method='normalize'
    dist.moment_standard(0)  # build up the cache
    dist.moment_standard(1)
    dist.moment_standard(2)
    for i in range(3, 6):
        ref = dist.moment_central(i, method='quadrature')
        check(dist.moment_central, i, 'normalize', ref,
              success=formula_standard)
        dist.moment_standard(i)  # build up the cache
        check(dist.moment_central, i, 'normalize', ref)

    dist._moment_raw_cache = {}
    dist._moment_central_cache = {}
    dist._moment_standard_cache = {}

    ### Check Standard Moments ###

    var = dist.moment_central(2, method='quadrature')
    del dist._moment_central_cache[2]

    for i in range(6):
        check(dist.moment_standard, i, 'cache', success=False)
        ref = dist.moment_central(i, method='quadrature') / var ** (i / 2)
        check_nans_and_edges(dist, 'moment_standard', None, ref)
        del dist._moment_central_cache[i]
        assert ref.shape == result_shape
        check(dist.moment_standard, i, 'formula', ref,
              success=formula_standard)
        check(dist.moment_standard, i, 'general', ref, success=i <= 2)
        check(dist.moment_standard, i, 'normalize', ref)

    ### Check Against _logmoment ###
    logmean = dist._logmoment(1, logcenter=-np.inf)
    for i in range(6):
        ref = np.exp(dist._logmoment(i, logcenter=-np.inf))
        assert_allclose(dist.moment_raw(i), ref, atol=atol)

        ref = np.exp(dist._logmoment(i, logcenter=logmean))
        assert_allclose(dist.moment_central(i), ref, atol=atol)

        ref = np.exp(dist._logmoment(i, logcenter=logmean, standardized=True))
        assert_allclose(dist.moment_standard(i), ref, atol=atol)


@pytest.mark.parametrize('family', (LogUniform, Normal))
@pytest.mark.parametrize('x_shape', [tuple(), (2, 3)])
@pytest.mark.parametrize('dist_shape', [tuple(), (4, 1)])
def test_sample_against_cdf(family, dist_shape, x_shape):
    rng = np.random.default_rng(842582438235635)
    num_parameters = family._num_parameters()

    if dist_shape and num_parameters == 0:
        pytest.skip("Distribution can't have a shape without parameters.")

    dist = family._draw(dist_shape, rng)

    n = 1000
    sample_size = (n,) + x_shape
    sample_array_shape = sample_size + dist_shape

    x = dist.sample(sample_size, rng=rng)
    assert x.shape == sample_array_shape

    # probably should give `axis` argument to ks_1samp, review that separately
    statistic = _kolmogorov_smirnov(dist, x, axis=0)
    pvalue = kolmogn(x.shape[0], statistic, cdf=False)
    p_threshold = 0.01
    num_pvalues = pvalue.size
    num_small_pvalues = np.sum(pvalue < p_threshold)
    assert num_small_pvalues < p_threshold * num_pvalues


def get_valid_parameters(dist):
    # Given a distribution, return a logical array that is true where all
    # distribution parameters are within their respective domains. The code
    # here is probably quite similar to that used to form the `_invalid`
    # attribute of the distribution, but this was written about a week later
    # without referring to that code, so it is a somewhat independent check.

    # Get all parameter values and `_Parameter` objects
    parameter_values = dist._parameters
    parameters = {}
    for parameterization in dist._parameterizations:
        parameters.update(parameterization.parameters)

    all_valid = np.ones(dist._shape, dtype=bool)
    for name, value in parameter_values.items():
        parameter = parameters[name]

        # Check that the numerical endpoints and inclusivity attribute
        # agree with the `contains` method about which parameter values are
        # within the domain.
        a, b = parameter.domain.get_numerical_endpoints(
            parameter_values=parameter_values)
        a_included, b_included = parameter.domain.inclusive
        valid = (a <= value) if a_included else a < value
        valid &= (value <= b) if b_included else value < b
        assert_equal(valid, parameter.domain.contains(
            value, parameter_values=parameter_values))

        # Form `all_valid` mask that is True where *all* parameters are valid
        all_valid &= valid

    # Check that the `all_valid` mask formed here is the complement of the
    # `dist._invalid` mask stored by the infrastructure
    assert_equal(~all_valid, dist._invalid)

    return all_valid

def classify_arg(dist, arg, arg_domain):
    if arg is None:
        valid_args = np.ones(dist._shape, dtype=bool)
        endpoint_args = np.zeros(dist._shape, dtype=bool)
        outside_args = np.zeros(dist._shape, dtype=bool)
        nan_args = np.zeros(dist._shape, dtype=bool)
        return valid_args, endpoint_args, outside_args, nan_args

    a, b = arg_domain.get_numerical_endpoints(
        parameter_values=dist._parameters)

    # # Not sure if this belongs here
    # adist, bdist = dist.support
    # assert_equal(a[~dist._invalid], adist[~dist._invalid])
    # assert_equal(b[~dist._invalid], bdist[~dist._invalid])
    # assert_equal(np.isnan(adist), dist._invalid)
    # assert_equal(np.isnan(bdist), dist._invalid)

    a, b, arg = np.broadcast_arrays(a, b, arg)
    a_included, b_included = arg_domain.inclusive

    inside = (a <= arg) if a_included else a < arg
    inside &= (arg <= b) if b_included else arg < b
    # TODO: add `supported` method and check here
    on = np.zeros(a.shape, dtype=int)
    on[a == arg] = -1
    on[b == arg] =1
    outside = np.zeros(a.shape, dtype=int)
    outside[(arg < a) if a_included else arg <= a] = -1
    outside[(b < arg) if b_included else b <= arg] = 1
    nan = np.isnan(arg)

    return inside, on, outside, nan
