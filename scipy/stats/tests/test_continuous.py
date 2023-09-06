import functools
import warnings
import pickle
from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from hypothesis import strategies, given, reproduce_failure  # noqa
import hypothesis.extra.numpy as npst

from scipy import stats
from scipy.stats._fit import _kolmogorov_smirnov
from scipy.stats._ksstats import kolmogn

from scipy.stats._distribution_infrastructure import (
    oo, _Domain, _RealDomain, _RealParameter, ContinuousDistribution,
    _Parameterization, ShiftedScaledDistribution)
from scipy.stats._new_distributions import LogUniform, Normal, ShiftedScaledNormal

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
        # dtype = np.result_type(a.dtype, b.dtype, x.dtype)
        # a, b, x = a.astype(dtype), b.astype(dtype), x.astype(dtype)
        # unclear whether we should be careful about this, since it will be
        # fixed with NEP50. Just do what makes the test pass.
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


def draw_distribution_from_family(family, data, rng, proportions, min_side=0):
    # If the distribution has parameters, choose a parameterization and
    # draw broadcastable shapes for the parameter arrays.
    n_parameterizations = family._num_parameterizations()
    if n_parameterizations > 0:
        i = data.draw(strategies.integers(0, max_value=n_parameterizations-1))
        n_parameters = family._num_parameters(i)
        shapes, result_shape = data.draw(
            npst.mutually_broadcastable_shapes(num_shapes=n_parameters,
                                               min_side=min_side))
        dist = family._draw(shapes, rng=rng, proportions=proportions,
                            i_parameterization=i)
    else:
        dist = family._draw(rng=rng)
        result_shape = tuple()

    # Draw a broadcastable shape for the arguments, and draw values for the
    # arguments.
    x_shape = data.draw(npst.broadcastable_shapes(result_shape,
                                                  min_side=min_side))
    x = dist._variable.draw(x_shape, parameter_values=dist._parameters,
                            proportions=proportions, rng=rng)
    x_result_shape = np.broadcast_shapes(x_shape, result_shape)
    y_shape = data.draw(npst.broadcastable_shapes(x_result_shape,
                                                  min_side=min_side))
    y = dist._variable.draw(y_shape, parameter_values=dist._parameters,
                            proportions=proportions, rng=rng)
    xy_result_shape = np.broadcast_shapes(y_shape, x_result_shape)
    p_domain = _RealDomain((0, 1), (True, True))
    p = p_domain.draw(x_shape, proportions=proportions, rng=rng)
    logp = np.log(p)

    return dist, x, y, p, logp, result_shape, x_result_shape, xy_result_shape


class TestDistributions:
    @pytest.mark.filterwarnings("ignore")
    # @pytest.mark.parametrize('family', (LogUniform,))
    # @pytest.mark.parametrize('family', (Normal,))
    # @pytest.mark.parametrize('family', (ShiftedScaledNormal,))
    @pytest.mark.parametrize('family', (Normal, LogUniform, ShiftedScaledNormal))
    @given(data=strategies.data(), seed=strategies.integers(min_value=0))
    def test_basic(self, family, data, seed):
        rng = np.random.default_rng(seed)

        # relative proportions of valid, endpoint, out of bounds, and NaN params
        proportions = (1, 1, 1, 1)
        tmp = draw_distribution_from_family(family, data, rng, proportions)
        dist, x, y, p, logp, result_shape, x_result_shape, xy_result_shape = tmp
        sample_shape = data.draw(npst.array_shapes(min_dims=0, min_side=0,
                                                   max_side=20))

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
        check_sample_shape_NaNs(dist, 'sample', sample_shape, result_shape)
        check_sample_shape_NaNs(dist, 'qmc_sample', sample_shape, result_shape)

        methods = {'log/exp'}
        check_dist_func(dist, 'pdf', x, x_result_shape, methods);
        check_dist_func(dist, 'logpdf', x, x_result_shape, methods)

        methods = {'log/exp', 'complementarity', 'quadrature'}
        check_dist_func(dist, 'logcdf', x, x_result_shape, methods)
        check_dist_func(dist, 'cdf', x, x_result_shape, methods)
        check_dist_func(dist, 'logccdf', x, x_result_shape, methods)
        check_dist_func(dist, 'ccdf', x, x_result_shape, methods)

        if not isinstance(dist, ShiftedScaledDistribution):
            methods = {'quadrature'}
            check_cdf2(dist, False, x, y, xy_result_shape, methods)
            check_cdf2(dist, True, x, y, xy_result_shape, methods)
            methods = {'addition'}
            check_ccdf2(dist, False, x, y, xy_result_shape, methods)
            check_ccdf2(dist, True, x, y, xy_result_shape, methods)

        methods = {'complementarity', 'inversion'}
        check_dist_func(dist, 'ilogcdf', logp, x_result_shape, methods)
        check_dist_func(dist, 'icdf', p, x_result_shape, methods)
        check_dist_func(dist, 'ilogccdf', logp, x_result_shape, methods)
        check_dist_func(dist, 'iccdf', p, x_result_shape, methods)


def check_sample_shape_NaNs(dist, fname, sample_shape, result_shape):
    full_shape = sample_shape + result_shape
    if fname == 'sample':
        sample_method = dist.sample
    elif fname == 'qmc_sample':
        sample_method = functools.partial(dist.sample,
                                          qmc_engine=stats.qmc.Halton)
    methods = {'inverse_transform'}
    if dist._overrides(f'_{fname}_formula'):
        methods.add('formula')

    for method in methods:
        res = sample_method(sample_shape, method=method)
        valid_parameters = np.broadcast_to(get_valid_parameters(dist),
                                           res.shape)
        assert_equal(res.shape, full_shape)
        np.testing.assert_equal(res.dtype, dist._dtype)

        if full_shape == ():
            # NumPy random makes a distinction between a 0d array and a scalar.
            # In stats, we consistently turn 0d arrays into scalars, so
            # maintain that behavior here. (With Array API arrays, this will
            # change.)
            assert np.isscalar(res)
        assert np.all(np.isfinite(res[valid_parameters]))
        assert_equal(res[~valid_parameters], np.nan)

        sample1 = sample_method(sample_shape, method=method,
                                rng=np.random.default_rng(42))
        sample2 = sample_method(sample_shape, method=method,
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

    if dist._overrides(f'_{fname}_formula'):
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

        # for now, make sure dtypes are consistent; later, we can check whether
        # they are correct.
        np.testing.assert_equal(res.dtype, ref.dtype)
        np.testing.assert_equal(res.shape, result_shape)
        if result_shape == tuple():
            assert np.isscalar(res)

def check_cdf2(dist, log, x, y, result_shape, methods):
    # Specialized test for 2-arg cdf since the interface is a bit different
    # from the other methods. Here, we'll use 1-arg cdf as a reference, and
    # since we have already checked 1-arg cdf in `check_nans_and_edges`, this
    # checks the equivalent of both `check_dist_func` and
    # `check_nans_and_edges`.
    methods = methods.copy()

    if log:
        if dist._overrides(f'_logcdf2_formula'):
            methods.add('formula')
        if dist._overrides(f'_logcdf_formula') or dist._overrides(f'_logccdf_formula'):
            methods.add('subtraction')
        if (dist._overrides(f'_cdf_formula')
                or dist._overrides(f'_ccdf_formula')):
            methods.add('log/exp')
    else:
        if dist._overrides(f'_cdf2_formula'):
            methods.add('formula')
        if dist._overrides(f'_cdf_formula') or dist._overrides(f'_ccdf_formula'):
            methods.add('subtraction')
        if (dist._overrides(f'_logcdf_formula')
                or dist._overrides(f'_logccdf_formula')):
            methods.add('log/exp')

    ref = dist.cdf(y) - dist.cdf(x)
    np.testing.assert_equal(ref.shape, result_shape)

    if result_shape == tuple():
        assert np.isscalar(ref)

    for method in methods:
        res = (np.exp(dist.logcdf(x, y, method=method)) if log
               else dist.cdf(x, y, method=method))
        np.testing.assert_allclose(res, ref, atol=1e-14)
        if log and np.any(x > y):
            np.testing.assert_equal(res.dtype, (ref + 0j).dtype)
        else:
            np.testing.assert_equal(res.dtype, ref.dtype)
        np.testing.assert_equal(res.shape, result_shape)
        if result_shape == tuple():
            assert np.isscalar(res)


def check_ccdf2(dist, log, x, y, result_shape, methods):
    # Specialized test for 2-arg ccdf since the interface is a bit different
    # from the other methods. Could be combined with check_cdf2 above, but
    # writing it separately is simpler.
    methods = methods.copy()

    if dist._overrides(f'_{"log" if log else ""}ccdf2_formula'):
        methods.add('formula')

    ref = dist.cdf(x) + dist.ccdf(y)
    np.testing.assert_equal(ref.shape, result_shape)

    if result_shape == tuple():
        assert np.isscalar(ref)

    for method in methods:
        res = (np.exp(dist.logccdf(x, y, method=method)) if log
               else dist.ccdf(x, y, method=method))
        np.testing.assert_allclose(res, ref, atol=1e-14)
        np.testing.assert_equal(res.dtype, ref.dtype)
        np.testing.assert_equal(res.shape, result_shape)
        if result_shape == tuple():
            assert np.isscalar(res)


def check_nans_and_edges(dist, fname, arg, res):

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

    a, b = dist.support()
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

    formula_raw = dist._overrides('_moment_raw_formula')
    formula_central = dist._overrides('_moment_central_formula')
    formula_standard = dist._overrides('_moment_standard_formula')
    dist.reset_cache()

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
    dist.reset_cache()

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

    dist.reset_cache()

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

    dist.reset_cache()

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

    ### Check Standard Moments ###

    var = dist.moment_central(2, method='quadrature')
    dist.reset_cache()

    for i in range(6):
        check(dist.moment_standard, i, 'cache', success=False)
        ref = dist.moment_central(i, method='quadrature') / var ** (i / 2)
        check_nans_and_edges(dist, 'moment_standard', None, ref)
        assert ref.shape == result_shape
        check(dist.moment_standard, i, 'formula', ref,
              success=formula_standard)
        check(dist.moment_standard, i, 'general', ref, success=i <= 2)
        check(dist.moment_standard, i, 'normalize', ref)

    if isinstance(dist, ShiftedScaledDistribution):
        # logmoment is not fully fleshed out; no need to test
        # ShiftedScaledDistribution here
        return

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
@pytest.mark.parametrize('fname', ['qmc_sample', 'sample'])
def test_sample_against_cdf(family, dist_shape, x_shape, fname):
    rng = np.random.default_rng(842582438235635)
    num_parameters = family._num_parameters()

    if dist_shape and num_parameters == 0:
        pytest.skip("Distribution can't have a shape without parameters.")

    dist = family._draw(dist_shape, rng)

    n = 1000
    sample_size = (n,) + x_shape
    sample_array_shape = sample_size + dist_shape

    if fname == 'sample':
        sample_method = dist.sample
    elif fname == 'qmc_sample':
        sample_method = functools.partial(dist.sample,
                                          qmc_engine=stats.qmc.Halton)
    x = sample_method(sample_size, rng=rng)
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
        if name not in parameters:  # cached value not part of parameterization
            continue
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
    # adist, bdist = dist.support()
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


def test_input_validation():
    class Test(ContinuousDistribution):
        _variable = _RealParameter('x', domain=_RealDomain())

    message = ("The `Test` distribution family does not accept parameters, "
               "but parameters `{'a'}` were provided.")
    with pytest.raises(ValueError, match=message):
        Test(a=1, )

    message = ("Attribute `tol` of `Test` must be a positive float, if "
               "specified.")
    with pytest.raises(ValueError, match=message):
        Test(tol=np.asarray([]))
    with pytest.raises(ValueError, match=message):
        Test(tol=[1, 2, 3])
    with pytest.raises(ValueError, match=message):
        Test(tol=np.nan)
    with pytest.raises(ValueError, match=message):
        Test(tol=-1)

    message = ("Argument `order` of `Test.moment_raw` must be a "
               "finite, positive integer.")
    with pytest.raises(ValueError, match=message):
        Test().moment_raw(-1)
    message = "Argument `order` of `Test.moment_central` must be..."
    with pytest.raises(ValueError, match=message):
        Test().moment_central(np.inf)
    message = "Argument `order` of `Test.moment_standard` must be..."
    with pytest.raises(ValueError, match=message):
        Test().moment_standard([1, 2, 3])

    message = ("Argument `rng` passed to the `Test` distribution family is of "
               "type `<class 'int'>`, but it must be a NumPy `Generator`.")
    with pytest.raises(ValueError, match=message):
        Test(rng=1)

    class Test2(ContinuousDistribution):
        _p1 = _RealParameter('c', domain=_RealDomain())
        _p2 = _RealParameter('d', domain=_RealDomain())
        _parameterizations = [_Parameterization(_p1, _p2)]
        _variable = _RealParameter('x', domain=_RealDomain())

    message = ("The provided parameters `{a}` do not match a supported "
               "parameterization of the `Test2` distribution family.")
    with pytest.raises(ValueError, match=message):
        Test2(a=1)
        
    message = ("The `Test2` distribution family requires parameters, but none "
               "were provided.")
    with pytest.raises(ValueError, match=message):
        Test2()

    message = ("The parameters `{c, d}` provided to the `Test2` "
               "distribution family cannot be broadcast to the same shape.")
    with pytest.raises(ValueError, match=message):
        Test2(c=[1, 2], d=[1, 2, 3])

    message = ("The argument provided to `Test2.pdf` cannot be be broadcast to "
              "the same shape as the distribution parameters.")
    with pytest.raises(ValueError, match=message):
        dist = Test2(c=[1, 2, 3], d=[1, 2, 3])
        dist.pdf([1, 2])

    message = ("Parameter `c` must be of real dtype.")
    with pytest.raises(ValueError, match=message):
        Test2(c=[1, object()], d=[1, 2])

# I removed `None` from this list. The current behavior is to generate a new
# `default_rng()` every time it is needed. We should not generate it during
# initialization because it increases the time by more than 50%!
@pytest.mark.parametrize('seed', [23434924629239023])
def test_deepcopy_pickle(seed):
    kwargs = dict(a=[-1, 2], b=10)
    if seed:
        kwargs['rng'] = np.random.default_rng(seed)
    dist1 = LogUniform(**kwargs)
    dist2 = deepcopy(dist1)
    dist3 = pickle.loads(pickle.dumps(dist1))
    res1, res2, res3 = dist1.sample(), dist2.sample(), dist3.sample()
    assert_equal(res2, res1)
    assert_equal(res3, res1)


def test_cache_policy():
    dist = Normal(cache_policy="no_cache")
    # make error message more appropriate
    message = "`Normal` does not provide an accurate implementation of the "
    with pytest.raises(NotImplementedError, match=message):
        dist.mean(method='cache')
    mean = dist.mean()
    with pytest.raises(NotImplementedError, match=message):
        dist.mean(method='cache')

    # add to enum
    dist.cache_policy = None
    with pytest.raises(NotImplementedError, match=message):
        dist.mean(method='cache')
    mean = dist.mean()  # method is 'formula' by default
    cached_mean = dist.mean(method='cache')
    assert_equal(cached_mean, mean)

    # cache is overridden by latest evaluation
    quadrature_mean = dist.mean(method='quadrature')
    cached_mean = dist.mean(method='cache')
    assert_equal(cached_mean, quadrature_mean)
    assert not np.all(mean == quadrature_mean)

    # We can turn the cache off, and it won't change, but the old cache is
    # still available
    dist.cache_policy = "no_cache"
    mean = dist.mean(method='formula')
    cached_mean = dist.mean(method='cache')
    assert_equal(cached_mean, quadrature_mean)
    assert not np.all(mean == quadrature_mean)

    dist.reset_cache()
    with pytest.raises(NotImplementedError, match=message):
        dist.mean(method='cache')


class TestTransforms:
    @pytest.mark.filterwarnings("ignore")
    @given(data=strategies.data(), seed=strategies.integers(min_value=0))
    def test_loc_scale(self, data, seed):
        rng = np.random.default_rng(seed)

        tmp = draw_distribution_from_family(
            ShiftedScaledNormal, data, rng, proportions=(1, 0, 0, 0), min_side=1)
        dist, x, y, p, logp, result_shape, x_result_shape, xy_result_shape = tmp
        sample_shape = data.draw(npst.array_shapes(min_dims=0, min_side=1,
                                                   max_side=20))
        loc = dist.loc
        scale = dist.scale
        dist0 = Normal()
        dist_ref = stats.norm(loc=loc, scale=scale)

        x0 = (x - loc) / scale

        assert_allclose(dist.logentropy(), np.log(dist.entropy() + 0j))
        assert_allclose(dist.entropy(), dist_ref.entropy())
        assert_allclose(dist.median(), dist0.median() + loc)
        assert_allclose(dist.mode(), dist0.mode() + loc)
        assert_allclose(dist.logpdf(x), dist0.logpdf(x0) - np.log(scale))
        assert_allclose(dist.pdf(x), dist0.pdf(x0) / scale)
        assert_allclose(dist.logcdf(x), dist0.logcdf(x0))
        assert_allclose(dist.cdf(x), dist0.cdf(x0))
        assert_allclose(dist.logccdf(x), dist0.logccdf(x0))
        assert_allclose(dist.ccdf(x), dist0.ccdf(x0))
        assert_allclose(dist.ilogcdf(logp), dist0.ilogcdf(logp)*scale + loc)
        assert_allclose(dist.icdf(p), dist0.icdf(p)*scale + loc)
        assert_allclose(dist.ilogccdf(logp), dist0.ilogccdf(logp)*scale + loc)
        assert_allclose(dist.iccdf(p), dist0.iccdf(p)*scale + loc)
        for i in range(1, 5):
            assert_allclose(dist.moment_raw(i), dist_ref.moment(i))
            assert_allclose(dist.moment_central(i),
                            dist0.moment_central(i)*scale**i)
            assert_allclose(dist.moment_standard(i),
                            dist0.moment_standard(i) * np.sign(scale) ** i)


        dist = (dist - 2*loc) + loc
        dist = dist/scale**2 * scale
        z = np.zeros(dist._shape)  # compact broadcasting

        assert_allclose(dist.logentropy(), dist0.logentropy() + z)
        assert_allclose(dist.entropy(), dist0.entropy() + z)
        assert_allclose(dist.median(), dist0.median() + z)
        assert_allclose(dist.mode(), dist0.mode() + z)
        assert_allclose(dist.logpdf(x), dist0.logpdf(x)+z)
        assert_allclose(dist.pdf(x), dist0.pdf(x) + z)
        assert_allclose(dist.logcdf(x), dist0.logcdf(x) + z)
        assert_allclose(dist.cdf(x), dist0.cdf(x) + z)
        assert_allclose(dist.logccdf(x), dist0.logccdf(x) + z)
        assert_allclose(dist.ccdf(x), dist0.ccdf(x) + z)
        assert_allclose(dist.ilogcdf(logp), dist0.ilogcdf(logp) + z)
        assert_allclose(dist.icdf(p), dist0.icdf(p) + z)
        assert_allclose(dist.ilogccdf(logp), dist0.ilogccdf(logp) + z)
        assert_allclose(dist.iccdf(p), dist0.iccdf(p) + z)
        for i in range(1, 5):
            assert_allclose(dist.moment_raw(i), dist0.moment_raw(i))
            assert_allclose(dist.moment_central(i), dist0.moment_central(i))
            assert_allclose(dist.moment_standard(i), dist0.moment_standard(i))

        # These are tough to compare because of the way the shape works
        # rng = np.random.default_rng(seed)
        # rng0 = np.random.default_rng(seed)
        # assert_allclose(dist.sample(x_result_shape, rng=rng),
        #                 dist0.sample(x_result_shape, rng=rng0) * scale + loc)
        # assert_allclose(dist.qmc_sample(x_result_shape, rng=rng),
        #                 dist0.qmc_sample(x_result_shape, rng=rng0) * scale + loc)
