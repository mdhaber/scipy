import warnings

import numpy as np
from .common import Benchmark, safe_import

with safe_import():
    import scipy.stats as stats
with safe_import():
    from scipy.stats._distr_params import distcont

try:  # builtin lib
    from itertools import compress
except ImportError:
    pass


class Anderson_KSamp(Benchmark):
    def setup(self, *args):
        self.rand = [np.random.normal(loc=i, size=1000) for i in range(3)]

    def time_anderson_ksamp(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            stats.anderson_ksamp(self.rand)


class CorrelationFunctions(Benchmark):
    param_names = ['alternative']
    params = [
        ['two-sided', 'less', 'greater']
    ]

    def setup(self, mode):
        a = np.random.rand(2,2) * 10
        self.a = a

    def time_fisher_exact(self, alternative):
        oddsratio, pvalue = stats.fisher_exact(self.a, alternative=alternative)


class InferentialStats(Benchmark):
    def setup(self):
        np.random.seed(12345678)
        self.a = stats.norm.rvs(loc=5, scale=10, size=500)
        self.b = stats.norm.rvs(loc=8, scale=10, size=20)
        self.c = stats.norm.rvs(loc=8, scale=20, size=20)

    def time_ttest_ind_same_var(self):
        # test different sized sample with variances
        stats.ttest_ind(self.a, self.b)
        stats.ttest_ind(self.a, self.b, equal_var=False)

    def time_ttest_ind_diff_var(self):
        # test different sized sample with different variances
        stats.ttest_ind(self.a, self.c)
        stats.ttest_ind(self.a, self.c, equal_var=False)


class Distribution(Benchmark):
    param_names = ['distribution', 'properties']
    params = [
        ['cauchy', 'gamma', 'beta'],
        ['pdf', 'cdf', 'rvs', 'fit']
    ]

    def setup(self, distribution, properties):
        np.random.seed(12345678)
        self.x = np.random.rand(100)

    def time_distribution(self, distribution, properties):
        if distribution == 'gamma':
            if properties == 'pdf':
                stats.gamma.pdf(self.x, a=5, loc=4, scale=10)
            elif properties == 'cdf':
                stats.gamma.cdf(self.x, a=5, loc=4, scale=10)
            elif properties == 'rvs':
                stats.gamma.rvs(size=1000, a=5, loc=4, scale=10)
            elif properties == 'fit':
                stats.gamma.fit(self.x, loc=4, scale=10)
        elif distribution == 'cauchy':
            if properties == 'pdf':
                stats.cauchy.pdf(self.x, loc=4, scale=10)
            elif properties == 'cdf':
                stats.cauchy.cdf(self.x, loc=4, scale=10)
            elif properties == 'rvs':
                stats.cauchy.rvs(size=1000, loc=4, scale=10)
            elif properties == 'fit':
                stats.cauchy.fit(self.x, loc=4, scale=10)
        elif distribution == 'beta':
            if properties == 'pdf':
                stats.beta.pdf(self.x, a=5, b=3, loc=4, scale=10)
            elif properties == 'cdf':
                stats.beta.cdf(self.x, a=5, b=3, loc=4, scale=10)
            elif properties == 'rvs':
                stats.beta.rvs(size=1000, a=5, b=3, loc=4, scale=10)
            elif properties == 'fit':
                stats.beta.fit(self.x, loc=4, scale=10)

    # Retain old benchmark results (remove this if changing the benchmark)
    time_distribution.version = "fb22ae5386501008d945783921fe44aef3f82c1dafc40cddfaccaeec38b792b0"

# 06a820d8df4f2e09d7e36267b5decda73b1d1a72 - ww
# c3e640c13c58bffa127eb8d3d4cae220d9c99c33 - master
class BetaDistribution(Benchmark):
    pass

class BetaDistributionSF(BetaDistribution):

    # Columns are x, a, b, cdf, sf.
    # The values of cdf and sf were computed using mpmath.
    _beta_cdf_sf = np.array([
        [1e-20, 3.0, 0.5, 3.1249999999999994e-61, 1.0],
        [1e-20, 0.5, 3.0, 1.875e-10, 0.9999999998125],
        [0.999, 2, 10, 1.0, 1.0990000000000097e-29],
        [0.9999999999999994, 3.0, 0.5, 0.9999999558234914, 4.417650858175538e-08],
        [0.9999999999999994, 0.5, 3.0, 1.0, 5.345529420184392e-47],
    ])

    param_names = ['params']
    params = _beta_cdf_sf

    def time_sf(self, params):
        x, a, b, cdf, sf = params
        stats.beta.sf(x, a=a, b=b)

    def track_sf(self, params):
        x, a, b, cdf, sf = params
        res = stats.beta.sf(x, a=a, b=b)
        return np.abs(sf - res)


class BetaDistributionISF(BetaDistribution):

    # Columns are p, a, b, isf.
    # The p column was computed using mpmath.
    _beta_isf = np.array([
        [0.99981250000125, 0.5, 3.0, 1.0e-8],
        [2.0000030000132007e-17, 0.5, 3.0, 0.999996],
        [5.930192380246625e-07, 3.0, 0.5, 0.9999999999999],
        [0.9996750253207289, 3.0, 0.5, 0.1],
    ])

    param_names = ['params']
    params = _beta_isf

    def time_isf(self, params):
        p, a, b, isf = params
        stats.beta.isf(p, a=a, b=b)

    def track_isf(self, params):
        p, a, b, isf = params
        res = stats.beta.isf(p, a=a, b=b)
        return np.abs(isf - res)


class DescriptiveStats(Benchmark):
    param_names = ['n_levels']
    params = [
        [10, 1000]
    ]

    def setup(self, n_levels):
        np.random.seed(12345678)
        self.levels = np.random.randint(n_levels, size=(1000, 10))

    def time_mode(self, n_levels):
        stats.mode(self.levels, axis=0)


class GaussianKDE(Benchmark):
    def setup(self):
        np.random.seed(12345678)
        n = 2000
        m1 = np.random.normal(size=n)
        m2 = np.random.normal(scale=0.5, size=n)

        xmin = m1.min()
        xmax = m1.max()
        ymin = m2.min()
        ymax = m2.max()

        X, Y = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
        self.positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([m1, m2])
        self.kernel = stats.gaussian_kde(values)

    def time_gaussian_kde_evaluate_few_points(self):
        # test gaussian_kde evaluate on a small number of points
        self.kernel(self.positions[:, :10])

    def time_gaussian_kde_evaluate_many_points(self):
        # test gaussian_kde evaluate on many points
        self.kernel(self.positions)


class GroupSampling(Benchmark):
    param_names = ['dim']
    params = [[3, 10, 50, 200]]

    def setup(self, dim):
        np.random.seed(12345678)

    def time_unitary_group(self, dim):
        stats.unitary_group.rvs(dim)

    def time_ortho_group(self, dim):
        stats.ortho_group.rvs(dim)

    def time_special_ortho_group(self, dim):
        stats.special_ortho_group.rvs(dim)


class BinnedStatisticDD(Benchmark):

    params = ["count", "sum", "mean", "min", "max", "median", "std", np.std]

    def setup(self, statistic):
        np.random.seed(12345678)
        self.inp = np.random.rand(9999).reshape(3, 3333) * 200
        self.subbin_x_edges = np.arange(0, 200, dtype=np.float32)
        self.subbin_y_edges = np.arange(0, 200, dtype=np.float64)
        self.ret = stats.binned_statistic_dd(
            [self.inp[0], self.inp[1]], self.inp[2], statistic=statistic,
            bins=[self.subbin_x_edges, self.subbin_y_edges])

    def time_binned_statistic_dd(self, statistic):
        stats.binned_statistic_dd(
            [self.inp[0], self.inp[1]], self.inp[2], statistic=statistic,
            bins=[self.subbin_x_edges, self.subbin_y_edges])

    def time_binned_statistic_dd_reuse_bin(self, statistic):
        stats.binned_statistic_dd(
            [self.inp[0], self.inp[1]], self.inp[2], statistic=statistic,
            binned_statistic_result=self.ret)


class ContinuousFitAnalyticalMLEOverride(Benchmark):
    # list of distributions to time
    dists = ["pareto", "laplace", "rayleigh",
             "invgauss", "gumbel_r", "gumbel_l"]
    # add custom values for rvs and fit, if desired, for any distribution:
    # key should match name in dists and value should be list of loc, scale,
    # and shapes
    custom_input = {}
    fnames = ['floc', 'fscale', 'f0', 'f1', 'f2']
    fixed = {}
    distcont = dict(distcont)

    param_names = ["distribution", "loc_fixed", "scale_fixed",
                   "shape1_fixed", "shape2_fixed", "shape3_fixed"]
    params = [dists, * [[True, False]] * 5]

    def setup(self, dist_name, loc_fixed, scale_fixed, shape1_fixed,
              shape2_fixed, shape3_fixed):
        self.distn = eval("stats." + dist_name)

        # default `loc` and `scale` are .834 and 4.342, and shapes are from
        # `_distr_params.py`
        default_shapes = self.distcont[dist_name]
        param_values = self.custom_input.get(dist_name, [.834, 4.342,
                                                         *default_shapes])
        # separate relevant and non-relevant parameters for this distribution
        # based on the number of shapes
        nparam = len(param_values)
        all_parameters = [loc_fixed, scale_fixed, shape1_fixed, shape2_fixed,
                          shape3_fixed]
        relevant_parameters = all_parameters[:nparam]
        nonrelevant_parameters = all_parameters[nparam:]

        # skip if all parameters are fixed or if non relevant parameters are
        # not all false
        if True in nonrelevant_parameters or False not in relevant_parameters:
            raise NotImplementedError("skip non-relevant case")

        # add fixed values if fixed in relevant_parameters to self.fixed
        # with keys from self.fnames and values from parameter_values
        self.fixed = dict(zip(compress(self.fnames, relevant_parameters),
                          compress(param_values, relevant_parameters)))
        self.data = self.distn.rvs(*param_values, size=1000)

    def time_fit(self, dist_name, loc_fixed, scale_fixed, shape1_fixed,
                 shape2_fixed, shape3_fixed):
        self.distn.fit(self.data, **self.fixed)
