import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from hypothesis import strategies, given, reproduce_failure  # noqa
import hypothesis.extra.numpy as npst

from scipy.stats._fit import _kolmogorov_smirnov
from scipy.stats._ksstats import kolmogn

from scipy.stats._distribution_infrastructure import (
    _Domain, _RealDomain, _RealParameter)
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

    @given(shapes=npst.mutually_broadcastable_shapes(num_shapes=3),
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


class TestDistributions:
    @pytest.mark.filterwarnings('ignore')  # remove this
    @pytest.mark.parametrize('family', (Normal, LogUniform,))
    @given(data=strategies.data())
    def test_basic(self, family, data):
        # strengthen this test by letting min_side=0 for both broadcasted shapes
        # check for scalar output if all inputs are scalar
        # inject bad parameters and x-values, check NaN pattern
        rng = np.random.default_rng(4826584632856)

        tmp = draw_distribution_from_family(family, data, rng)
        dist, x, p, logp, result_shape, x_result_shape = tmp

        methods = {'log/exp', 'quadrature'}
        check_dist_func(dist, 'entropy', None, result_shape, methods)
        check_dist_func(dist, 'logentropy', None, result_shape, methods)

        check_moment_funcs(dist, result_shape)

        methods = {'icdf'}
        check_dist_func(dist, 'median', None, result_shape, methods)

        methods = {'log/exp', 'logmoment'}
        check_dist_func(dist, 'logmean', None, result_shape, methods)
        check_dist_func(dist, 'logvar', None, result_shape, methods)
        check_dist_func(dist, 'logskewness', None, result_shape, methods)
        check_dist_func(dist, 'logkurtosis', None, result_shape, methods)

        methods = {'cache'}  #  weak test right now
        check_dist_func(dist, 'mean', None, result_shape, methods)
        check_dist_func(dist, 'var', None, result_shape, methods)
        check_dist_func(dist, 'skewness', None, result_shape, methods)
        check_dist_func(dist, 'kurtosis', None, result_shape, methods)

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


def draw_distribution_from_family(family, data, rng):
    # If the distribution has parameters, choose a parameterization and
    # draw broadcastable shapes for the parameter arrays.
    n_parameterizations = len(family._parameterizations)
    if n_parameterizations > 0:
        i_parameterization = (
            data.draw(strategies.integers(min_value=0,
                                          max_value=n_parameterizations - 1)))
        n_parameters = len(
            family._parameterizations[i_parameterization].parameters)
        shapes, result_shape = data.draw(
            npst.mutually_broadcastable_shapes(num_shapes=n_parameters))
        dist = family._draw(shapes, rng=rng,
                            i_parameterization=i_parameterization)
    else:
        dist = family._draw(rng=rng)
        result_shape = tuple()

    # Draw a broadcastable shape for the arguments, and draw values for the
    # arguments.
    x_shape = data.draw(npst.broadcastable_shapes(result_shape))
    x = dist._variable.draw(x_shape, parameter_values=dist._all_parameters)
    x_result_shape = np.broadcast_shapes(x_shape, result_shape)
    p = rng.uniform(size=x_shape)
    logp = np.log(p)

    return dist, x, p, logp, result_shape, x_result_shape


def check_dist_func(dist, fname, arg, result_shape, methods):
    # Check that all computation methods of all distribution functions agree
    # with one another, effectively testing the correctness of the generic
    # computation methods and confirming the consistency of specific
    # distributions with their pdf/logpdf.
    args = tuple() if arg is None else (arg,)
    ref = getattr(dist, fname)(*args)

    methods = methods.copy()

    tol_override = {}

    # Mean can be 0, which makes logmean -oo.
    if fname in {'logmean', 'mean', 'logskewness', 'skewness'}:
        tol_override = {'atol': 1e-15}
    else:
        assert np.isfinite(ref).all()

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

def check_moment_funcs(dist, result_shape):
    # Check that all computation methods of all distribution functions agree
    # with one another, effectively testing the correctness of the generic
    # computation methods and confirming the consistency of specific
    # distributions with their pdf/logpdf.

    def check(moment, order, method=None, ref=None, success=True):
        if success:
            res = moment(order, method=method)
            assert_allclose(res, ref, atol=1e-13)
            assert res.shape == ref.shape
        else:
            with pytest.raises(NotImplementedError):
                moment(order, method=method)

    formula_raw = dist._overrides('_moment_raw')
    formula_central = dist._overrides('_moment_central')
    formula_standard = dist._overrides('_moment_standard')

    ### Check Raw Moments ###
    for i in range(6):
        check(dist.moment_raw, i, 'cache', success=False)  # not cached yet
        ref = dist.moment_raw(i, method='quadrature')
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
        del dist._moment_central_cache[i]
        assert ref.shape == result_shape
        check(dist.moment_standard, i, 'formula', ref,
              success=formula_standard)
        check(dist.moment_standard, i, 'general', ref, success=i <= 2)
        check(dist.moment_standard, i, 'normalize', ref)


@pytest.mark.parametrize('family', (LogUniform, Normal))
@pytest.mark.parametrize('x_shape', [tuple(), (2, 3)])
@pytest.mark.parametrize('dist_shape', [tuple(), (4, 1)])
def test_sample(family, dist_shape, x_shape):
    rng = np.random.default_rng(842582438235635)
    num_parameterizations = family._num_parameterizations()
    num_parameters = family._num_parameters()

    if dist_shape and num_parameters == 0:
        pytest.skip("Distribution can't have a shape without parameters.")

    i = rng.integers(0, max(0, num_parameterizations-1), endpoint=True)
    dist = family._draw([dist_shape]*num_parameters, rng, i_parameterization=i)

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
