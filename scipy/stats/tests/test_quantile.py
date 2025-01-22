import pytest
import numpy as np
from scipy import stats
from scipy._lib._array_api_no_0d import xp_assert_close, xp_assert_equal
from scipy._lib._util import _apply_over_batch

skip_xp_backends = pytest.mark.skip_xp_backends


@_apply_over_batch(('x', 1), ('p', 1))
def quantile_reference_last_axis(x, p, nan_policy, method):
    if nan_policy == 'omit':
        x = x[~np.isnan(x)]
    p_mask = np.isnan(p)
    p = p.copy()
    p[p_mask] = 0.5
    if method == 'harrell-davis':
        # hdquantiles returns masked element if length along axis is 1 (bug)
        res = (np.full_like(p, x[0]) if x.size == 1
               else stats.mstats.hdquantiles(x, p).data)
        if nan_policy == 'propagate' and np.any(np.isnan(x)):
            res[:] = np.nan
    else:
        res = np.quantile(x, p)
    res[p_mask] = np.nan
    return res


def quantile_reference(x, p, *, axis, nan_policy, keepdims, method):
    x, p = np.moveaxis(x, axis, -1), np.moveaxis(p, axis, -1)
    res = quantile_reference_last_axis(x, p, nan_policy, method)
    res = np.moveaxis(res, -1, axis)
    if not keepdims:
        res = np.squeeze(res, axis=axis)
    return res


@skip_xp_backends('array_api_strict', reason="No take_along_axis yet.")
@skip_xp_backends('jax.numpy', reason="No mutation.")
class TestQuantile:

    def test_input_validation(self, xp):
        x = xp.asarray([1, 2, 3])
        p = xp.asarray(0.5)

        message = "`x` must have real dtype."
        with pytest.raises(ValueError, match=message):
            stats.quantile(xp.asarray([True, False]), p)
        with pytest.raises(ValueError):
            stats.quantile(xp.asarray([1+1j, 2]), p)

        message = "`p` must have real floating dtype."
        with pytest.raises(ValueError, match=message):
            stats.quantile(x, xp.asarray([0, 1]))

        message = "`axis` must be an integer or None."
        with pytest.raises(ValueError, match=message):
            stats.quantile(x, p, axis=0.5)
        with pytest.raises(ValueError, match=message):
            stats.quantile(x, p, axis=(0, -1))

        message = "`axis` is not compatible with the shapes of the inputs."
        with pytest.raises(ValueError, match=message):
            stats.quantile(x, p, axis=2)

        message = "The input contains nan values"
        with pytest.raises(ValueError, match=message):
            stats.quantile(xp.asarray([xp.nan, 1, 2]), p, nan_policy='raise')

        message = "method` must be one of..."
        with pytest.raises(ValueError, match=message):
            stats.quantile(x, p, method='a duck')

        message = "If specified, `keepdims` must be True or False."
        with pytest.raises(ValueError, match=message):
            stats.quantile(x, p, keepdims=42)

        message = "`keepdims` may be False only if the length of `p` along `axis` is 1."
        with pytest.raises(ValueError, match=message):
            stats.quantile(x, xp.asarray([0.5, 0.6]), keepdims=False)


    @pytest.mark.parametrize('method',
                             ['hazen', 'interpolated_inverted_cdf', 'linear',
                              'median_unbiased', 'normal_unbiased', 'weibull'])
    @pytest.mark.parametrize('shape_x, shape_p, axis',
                             [(10, None, -1), (10, 3, -1), (10, (2, 3), -1),
                              ((10, 2), None, 0), ((10, 2), None, 0)])
    def test_against_numpy(self, method, shape_x, shape_p, axis, xp):
        rng = np.random.default_rng(23458924568734956)
        x = rng.random(size=shape_x)
        p = rng.random(size=shape_p)
        ref = np.quantile(x, p, method=method, axis=axis)

        x, p = xp.asarray(x), xp.asarray(p)
        res = stats.quantile(xp.asarray(x), xp.asarray(p), method=method, axis=axis)

        xp_assert_close(res, xp.asarray(ref))


    @pytest.mark.parametrize('axis', [0, 1])
    @pytest.mark.parametrize('keepdims', [False, True])
    @pytest.mark.parametrize('nan_policy', ['omit', 'propagate'])
    @pytest.mark.parametrize('dtype', ['float32', 'float64'])
    @pytest.mark.parametrize('method', ['linear', 'harrell-davis'])
    def test_against_reference(self, axis, keepdims, nan_policy, dtype, method, xp):
        rng = np.random.default_rng(23458924568734956)
        shape = (3, 4)
        x = rng.random(size=shape).astype(dtype)
        p = rng.random(size=shape).astype(dtype)
        mask = rng.random(size=shape) > 0.9
        assert np.any(mask)
        x[mask] = np.nan
        if not keepdims:
            p = np.mean(p, axis=axis, keepdims=True)

        dtype = getattr(xp, dtype)
        kwargs = dict(axis=axis, keepdims=keepdims,
                      nan_policy=nan_policy, method=method)
        res = stats.quantile(xp.asarray(x), xp.asarray(p), **kwargs)
        ref = quantile_reference(x, p, **kwargs)
        xp_assert_close(res, xp.asarray(ref, dtype=dtype))


    @pytest.mark.parametrize('x, p, ref, kwargs',
        [([], 0.5, np.nan, {}),
         ([1, 2, 3], [-1, 0, 1, 1.5, np.nan], [np.nan, 1, 3, np.nan, np.nan], {}),
         ([1, 2, 3], [], [], {}),
         ([[np.nan, 2]], 0.5, [np.nan, 2], {'nan_policy': 'omit'}),
         ([[], []], 0.5, np.full(2, np.nan), {'axis': -1}),
         ([[], []], 0.5, np.zeros((0,)), {'axis': 0, 'keepdims': False}),
         ([[], []], 0.5, np.zeros((1, 0)), {'axis': 0, 'keepdims': True}),
         ([], [0.5, 0.6], np.full(2, np.nan), {}),])
    def test_edge_cases(self, x, p, ref, kwargs, xp):
        x, p, ref = xp.asarray(x), xp.asarray(p), xp.asarray(ref)
        res = stats.quantile(x, p, **kwargs)
        xp_assert_equal(res, ref)
