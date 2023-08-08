import warnings

import numpy as np
import pytest
import hypothesis
from numpy.testing import assert_allclose, assert_equal
from hypothesis import strategies, given, assume
import hypothesis.extra.numpy as npst
from numpy import ComplexWarning

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
        domain.define_parameters(_RealParameter('a'), _RealParameter('b'))
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
    @pytest.mark.filterwarnings('ignore')
    @pytest.mark.parametrize('family', (Normal, LogUniform))
    @given(data=strategies.data())
    def test_distribution(self, family, data):
        # strengthen this test by letting min_side=0 for both broadcasted shapes
        # check for scalar output if all inputs are scalar
        # inject bad parameters and x-values, check NaN pattern
        rng = np.random.default_rng(4826584632856)

        n_parameterizations = len(family._parameterizations)
        if n_parameterizations > 0:
            i_parameterization = data.draw(strategies.integers(min_value=0, max_value=n_parameterizations-1))
            n_parameters = len(family._parameterizations[i_parameterization].parameters)
            shapes, result_shape = data.draw(npst.mutually_broadcastable_shapes(num_shapes=n_parameters))
            dist = family._draw(shapes, rng=rng, i_parameterization=i_parameterization)
        else:
            dist = family._draw(rng=rng)
            result_shape = tuple()

        params = dist._all_shapes

        methods = {'log/exp', 'quadrature'}
        if dist._overrides('_logentropy'):
            methods.add('direct')
        ref = dist.logentropy()
        np.testing.assert_equal(ref.shape, result_shape)
        for method in methods:
            res = dist.logentropy(method=method)
            np.testing.assert_allclose(res, ref)
            assert np.isfinite(res).all()
            assert np.isfinite(ref).all()
            np.testing.assert_equal(res.shape, result_shape)

        methods = {'log/exp', 'quadrature'}
        if dist._overrides('_entropy'):
            methods.add('direct')
        ref = dist.entropy()
        np.testing.assert_equal(ref.shape, result_shape)
        for method in methods:
            res = dist.entropy(method=method)
            np.testing.assert_allclose(res, ref)
            assert np.isfinite(res).all()
            assert np.isfinite(ref).all()
            np.testing.assert_equal(res.shape, result_shape)

        shape = data.draw(npst.broadcastable_shapes(result_shape))
        x = dist._variable.draw(shape, shapes=dist._all_shapes)
        result_shape = np.broadcast_shapes(shape, result_shape)

        methods = {'log/exp'}
        if dist._overrides('_logpdf'):
            methods.add('direct')
        ref = dist.logpdf(x)
        np.testing.assert_equal(ref.shape, result_shape)
        for method in methods:
            res = dist.logpdf(x, method=method)
            np.testing.assert_allclose(res, ref)
            assert np.isfinite(res).all()
            assert np.isfinite(ref).all()
            np.testing.assert_equal(res.shape, result_shape)

        methods = {'log/exp'}
        if dist._overrides('_pdf'):
            methods.add('direct')
        ref = dist.pdf(x)
        np.testing.assert_equal(ref.shape, result_shape)
        for method in methods:
            res = dist.pdf(x, method=method)
            np.testing.assert_allclose(res, ref)
            assert np.isfinite(res).all()
            assert np.isfinite(ref).all()
            np.testing.assert_equal(res.shape, result_shape)

        methods = {'log/exp', 'complement', 'quadrature'}
        if dist._overrides('_logcdf'):
            methods.add('direct')
        ref = dist.logcdf(x)
        np.testing.assert_equal(ref.shape, result_shape)
        for method in methods:
            res = dist.logcdf(x, method=method)
            # the error of the CDF - not the logcdf - is controlled
            np.testing.assert_allclose(np.exp(res), np.exp(ref))
            assert np.isfinite(res).all()
            assert np.isfinite(ref).all()
            np.testing.assert_equal(res.shape, result_shape)

        methods = {'log/exp', 'complement', 'quadrature'}
        if dist._overrides('_cdf'):
            methods.add('direct')
        ref = dist.cdf(x)
        np.testing.assert_equal(ref.shape, result_shape)
        for method in methods:
            res = dist.cdf(x, method=method)
            np.testing.assert_allclose(res, ref)
            assert np.isfinite(res).all()
            assert np.isfinite(ref).all()
            np.testing.assert_equal(res.shape, result_shape)

        methods = {'log/exp', 'complement', 'quadrature'}
        if dist._overrides('_logccdf'):
            methods.add('direct')
        ref = dist.logccdf(x)
        np.testing.assert_equal(ref.shape, result_shape)
        for method in methods:
            res = dist.logccdf(x, method=method)
            np.testing.assert_allclose(np.exp(res), np.exp(ref))
            assert np.isfinite(res).all()
            assert np.isfinite(ref).all()
            np.testing.assert_equal(res.shape, result_shape)

        methods = {'log/exp', 'complement', 'quadrature'}
        if dist._overrides('_ccdf'):
            methods.add('direct')
        ref = dist.ccdf(x)
        np.testing.assert_equal(ref.shape, result_shape)
        for method in methods:
            res = dist.ccdf(x, method=method)
            np.testing.assert_allclose(res, ref)
            assert np.isfinite(res).all()
            assert np.isfinite(ref).all()
            np.testing.assert_equal(res.shape, result_shape)
