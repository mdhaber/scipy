from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from scipy import stats
from scipy.stats._qmc import check_random_state
from scipy.stats._stats_py import _var

if TYPE_CHECKING:
    import numpy.typing as npt
    from scipy._lib._util import SeedType
    from typing import Literal, Sequence  # noqa: UP035


__all__ = [
    'dunnett', 'false_discovery_control'
]


@dataclass
class DunnettResult:
    statistic: np.ndarray
    pvalue: np.ndarray


def dunnett(
    *samples: npt.ArrayLike,
    control: npt.ArrayLike,
    alternative: Literal['two-sided', 'less', 'greater'] = "two-sided",
    random_state: SeedType = None
) -> DunnettResult:
    """Dunnett's test: multiple comparisons of means against a control group.

    This is an implementation of Dunnett's original, single-step test as
    described in [1]_.

    Parameters
    ----------
    sample1, sample2, ... : 1D array_like
        The sample measurements for each experimental group.
    control : 1D array_like
        The sample measurements for the control group.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.

        The null hypothesis is that the means of the distributions underlying
        the samples and control are equal. The following alternative
        hypotheses are available (default is 'two-sided'):

        * 'two-sided': the means of the distributions underlying the samples
          and control are unequal.
        * 'less': the means of the distributions underlying the samples
          are less than the mean of the distribution underlying the control.
        * 'greater': the means of the distributions underlying the
          samples are greater than the mean of the distribution underlying
          the control.
    random_state : {None, int, `numpy.random.Generator`}, optional
        If `random_state` is an int or None, a new `numpy.random.Generator` is
        created using ``np.random.default_rng(random_state)``.
        If `random_state` is already a ``Generator`` instance, then the
        provided instance is used.

        The random number generator is used to control the randomized
        Quasi-Monte Carlo integration of the multivariate-t distribution.

    Returns
    -------
    res : DunnettResult
        An object containing attributes:

        statistic : float ndarray
            The computed statistic of the test for each comparison. The element
            at index ``i`` is the statistic for the comparison between
            groups ``i`` and the control.
        pvalue : float ndarray
            The computed p-value of the test for each comparison. The element
            at index ``i`` is the p-value for the comparison between
            group ``i`` and the control.

    See Also
    --------
    tukey_hsd : performs pairwise comparison of means.

    Notes
    -----
    Like the independent-sample t-test, Dunnett's test [1]_ is used to make
    inferences about the means of distributions from which samples were drawn.
    However, when multiple t-tests are performed at a fixed significance level,
    the "family-wise error rate" - the probability of incorrectly rejecting the
    null hypothesis in at least one test - will exceed the significance level.
    Dunnett's test is designed to perform multiple comparisons while
    controlling the family-wise error rate.

    Dunnett's test compares the means of multiple experimental groups
    against a single control group. Tukey's Honestly Significant Difference Test
    is another multiple-comparison test that controls the family-wise error
    rate, but `tukey_hsd` performs *all* pairwise comparisons between groups.
    When pairwise comparisons between experimental groups are not needed,
    Dunnett's test is preferable due to its higher power.


    The use of this test relies on several assumptions.

    1. The observations are independent within and among groups.
    2. The observations within each group are normally distributed.
    3. The distributions from which the samples are drawn have the same finite
       variance.

    References
    ----------
    .. [1] Charles W. Dunnett. "A Multiple Comparison Procedure for Comparing
       Several Treatments with a Control."
       Journal of the American Statistical Association, 50:272, 1096-1121,
       :doi:`10.1080/01621459.1955.10501294`, 1955.

    Examples
    --------
    In [1]_, the influence of drugs on blood count measurements on three groups
    of animal is investigated.

    The following table summarizes the results of the experiment in which
    two groups received different drugs, and one group acted as a control.
    Blood counts (in millions of cells per cubic millimeter) were recorded::

    >>> import numpy as np
    >>> control = np.array([7.40, 8.50, 7.20, 8.24, 9.84, 8.32])
    >>> drug_a = np.array([9.76, 8.80, 7.68, 9.36])
    >>> drug_b = np.array([12.80, 9.68, 12.16, 9.20, 10.55])

    We would like to see if the means between any of the groups are
    significantly different. First, visually examine a box and whisker plot.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(1, 1)
    >>> ax.boxplot([control, drug_a, drug_b])
    >>> ax.set_xticklabels(["Control", "Drug A", "Drug B"])  # doctest: +SKIP
    >>> ax.set_ylabel("mean")  # doctest: +SKIP
    >>> plt.show()

    Note the overlapping interquartile ranges of the drug A group and control
    group and the apparent separation between the drug B group and control
    group.

    Next, we will use Dunnett's test to assess whether the difference
    between group means is significant while controlling the family-wise error
    rate: the probability of making any false discoveries.
    Let the null hypothesis be that the experimental groups have the same
    mean as the control and the alternative be that an experimental group does
    not have the same mean as the control. We will consider a 5% family-wise
    error rate to be acceptable, and therefore we choose 0.05 as the threshold
    for significance.

    >>> from scipy.stats import dunnett
    >>> res = dunnett(drug_a, drug_b, control=control)
    >>> res.pvalue
    array([0.62004941, 0.0059035 ])  # may vary

    The p-value corresponding with the comparison between group A and control
    exceeds 0.05, so we do not reject the null hypothesis for that comparison.
    However, the p-value corresponding with the comparison between group B
    and control is less than 0.05, so we consider the experimental results
    to be evidence against the null hypothesis in favor of the alternative:
    group B has a different mean than the control group.

    """
    samples_, control_, rng = _iv_dunnett(
        samples=samples, control=control,
        alternative=alternative, random_state=random_state
    )

    rho, df, n_group = _params_dunnett(samples=samples_, control=control_)

    statistic = _statistic_dunnett(samples_, control_, df)

    pvalue = _pvalue_dunnett(
        rho=rho, df=df, statistic=statistic, alternative=alternative, rng=rng
    )

    return DunnettResult(
        statistic=statistic, pvalue=pvalue,
    )


def _iv_dunnett(
    samples: Sequence[npt.ArrayLike],
    control: npt.ArrayLike,
    alternative: Literal['two-sided', 'less', 'greater'],
    random_state: SeedType
) -> tuple[list[np.ndarray], np.ndarray, SeedType]:
    """Input validation for Dunnett's test."""
    rng = check_random_state(random_state)

    if alternative not in {'two-sided', 'less', 'greater'}:
        raise ValueError(
            "alternative must be 'less', 'greater' or 'two-sided'"
        )

    ndim_msg = "Control and samples groups must be 1D arrays"
    n_obs_msg = "Control and samples groups must have at least 1 observation"

    control = np.asarray(control)
    samples_ = [np.asarray(sample) for sample in samples]

    # samples checks
    samples_control: list[np.ndarray] = samples_ + [control]
    for sample in samples_control:
        if sample.ndim > 1:
            raise ValueError(ndim_msg)

        if sample.size < 1:
            raise ValueError(n_obs_msg)

    return samples_, control, rng


def _params_dunnett(
    samples: list[np.ndarray], control: np.ndarray
) -> tuple[np.ndarray, int, int]:
    """Specific parameters for Dunnett's test.

    Degree of freedom is the number of observations minus the number of groups
    including the control.
    """
    n_samples = np.array([sample.size for sample in samples])

    # From [1] p. 1100 d.f. = (sum N)-(p+1)
    n_sample = n_samples.sum()
    n_control = control.size
    n = n_sample + n_control
    n_groups = len(samples)
    df = n - n_groups - 1

    # From [1] p. 1103 rho_ij = 1/sqrt((N0/Ni+1)(N0/Nj+1))
    rho = n_control/n_samples + 1
    rho = 1/np.sqrt(rho[:, None] * rho[None, :])
    np.fill_diagonal(rho, 1)

    return rho, df, n_groups


def _statistic_dunnett(
    samples: list[np.ndarray], control: np.ndarray, df: int
) -> np.ndarray:
    """Statistic of Dunnett's test.

    Computation based on the original single-step test from [1].
    """
    n_control = control.size
    n_samples = np.array([sample.size for sample in samples])
    mean_control = np.mean(control)
    mean_samples = np.array([np.mean(sample) for sample in samples])
    all_samples = [control] + samples
    all_means = np.concatenate([[mean_control], mean_samples])

    # Variance estimate s^2 from [1] Eq. 1
    s2 = np.sum([_var(sample, mean=mean)*sample.size
                 for sample, mean in zip(all_samples, all_means)]) / df

    # z score inferred from [1] unlabeled equation after Eq. 1
    z = (mean_samples - mean_control) / np.sqrt(1/n_samples + 1/n_control)

    return z / np.sqrt(s2)


def _pvalue_dunnett(
    rho: np.ndarray, df: int, statistic: np.ndarray,
    alternative: Literal['two-sided', 'less', 'greater'],
    rng: SeedType = None
) -> np.ndarray:
    """pvalue from the multivariate t-distribution.

    Critical values come from the multivariate student-t distribution.
    """
    statistic = statistic.reshape(-1, 1)

    mvt = stats.multivariate_t(shape=rho, df=df, seed=rng)
    if alternative == "two-sided":
        statistic = abs(statistic)
        pvalue = 1 - mvt.cdf(statistic, lower_limit=-statistic)
    elif alternative == "greater":
        pvalue = 1 - mvt.cdf(statistic, lower_limit=-np.inf)
    else:
        pvalue = 1 - mvt.cdf(np.inf, lower_limit=statistic)

    return pvalue


def false_discovery_control(ps, *, axis=0, method='bh'):
    """Adjust p-values to control the false discovery rate.

    The false discovery rate (FDR) is the expected proportion of rejected null
    hypotheses that are actually true.
    If the null hypothesis is rejected when the *adjusted* p-value falls below
    a specified level, the false discovery rate is controlled at that level.

    Parameters
    ----------
    ps : 1D array_like
        The p-values to adjust. Elements must be real numbers between 0 and 1.
    axis : int
        The axis along which to perform the adjustment. The adjustment is
        performed independently along each axis-slice. If `axis` is None, `ps`
        is raveled before performing the adjustment.
    method : {'bh', 'by'}
        The false discovery rate control procedure to apply: ``'bh'`` is for
        Benjamini-Hochberg [1]_ (Eq. 1), ``'by'`` is for Benjaminini-Yekutieli
        [2]_ (Theorem 1.3). The latter is more conservative, but it is
        guaranteed to control the FDR even when the p-values are not from
        independent tests.

    Returns
    -------
    ps_adusted : array_like
        The adjusted p-values. If the null hypothesis is rejected where these
        fall below a specified level, the false discovery rate is controlled
        at that level.

    See Also
    --------
    combine_pvalues
    statsmodels.stats.multitest.multipletests

    Notes
    -----
    In multiple hypothesis testing, false discovery control procedures tend to
    offer higher power than familywise error rate control procedures (e.g.
    Bonferroni correction [1]_).

    If the p-values correspond with independent tests (or tests with
    "positive regression dependencies" [2]_), rejecting null hypotheses
    corresponding with Benjamini-Hochberg-adjusted p-values below :math:`q`
    controls the false discovery rate at a level less than or equal to
    :math:`q m_0 / m`, where :math:`m_0` is the number of true null hypotheses
    and :math:`m` is the total number of null hypotheses tested. The same is
    true even for dependent tests when the p-values are adjusted accorded to
    the more conservative Benjaminini-Yekutieli procedure.

    The adjusted p-values produced by this function are comparable to those
    produced by the R function ``p.adjust`` and the statsmodels function
    `statsmodels.stats.multitest.multipletests`. Please consider the latter
    for more advanced methods of multiple comparison correction.

    References
    ----------
    .. [1] Benjamini, Yoav, and Yosef Hochberg. "Controlling the false
           discovery rate: a practical and powerful approach to multiple
           testing." Journal of the Royal statistical society: series B
           (Methodological) 57.1 (1995): 289-300.

    .. [2] Benjamini, Yoav, and Daniel Yekutieli. "The control of the false
           discovery rate in multiple testing under dependency." Annals of
           statistics (2001): 1165-1188.

    .. [3] TileStats. FDR - Benjamini-Hochberg explained - Youtube.
           https://www.youtube.com/watch?v=rZKa4tW2NKs.

    .. [4] Neuhaus, Karl-Ludwig, et al. "Improved thrombolysis in acute
           myocardial infarction with front-loaded administration of alteplase:
           results of the rt-PA-APSAC patency study (TAPS)." Journal of the
           American College of Cardiology 19.5 (1992): 885-891.

    Examples
    --------
    We follow the example from [1]_.

        Thrombolysis with recombinant tissue-type plasminogen activator (rt-PA)
        and anisoylated plasminogen streptokinase activator (APSAC) in
        myocardial infarction has been proved to reduce mortality. [4]_
        investigated the effects of a new front-loaded administration of rt-PA
        versus those obtained with a standard regimen of APSAC, in a randomized
        multicentre trial in 421 patients with acute myocardial infarction.

    There were four families of hypotheses tested in the study, the last of
    which was "cardiac and other events after the start of thrombolitic
    treatment". FDR control may be desired in this family of hypotheses
    because it would not be appropriate to conclude that the front-loaded
    treatment is better if it is merely equivalent to the previous treatment.

    The p-values corresponding with the 15 hypotheses in this family were

    >>> ps = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344,
    ...       0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.000]

    If the chosen significance level is 0.05, we may be tempted to reject the
    null hypotheses for the tests corresponding with the first nine p-values,
    as the first nine p-values fall below the chosen significance level.
    However, this would ignore the problem of "multiplicity": if we fail to
    correct for the fact that multiple comparisons are being performed, we
    are more likely to incorrectly reject true null hypotheses.

    One approach to the multiplicity problem is to control the family-wise
    error rate (FWER), that is, the rate at which the null hypothesis is
    rejected when it is actually true. A common procedure of this kind is the
    Bonferroni correction [1]_.  We begin by multiplying the p-values by the
    number of hypotheses tested.

    >>> import numpy as np
    >>> np.array(ps) * len(ps)
    array([1.5000e-03, 6.0000e-03, 2.8500e-02, 1.4250e-01, 3.0150e-01,
           4.1700e-01, 4.4700e-01, 5.1600e-01, 6.8850e-01, 4.8600e+00,
           6.3930e+00, 8.5785e+00, 9.7920e+00, 1.1385e+01, 1.5000e+01])

    To control the FWER at 5%, we reject only the hypotheses corresponding
    with adjusted p-values less than 0.05. In this case, only the hypotheses
    corresponding with the first three p-values can be rejected. According to
    [1]_, these three hypotheses concerned "allergic reaction" and "two
    different aspects of bleeding."

    An alternative approach is to control the false discovery rate: the
    expected fraction of rejected null hypotheses that are actually true. The
    advantage of this approach is that it typically affords greater power: an
    increased rate of rejecting the null hypothesis when it is indeed false. To
    control the false discovery rate at 5%, we apply the Benjamini-Hochberg
    p-value adjustment.

    >>> from scipy import stats
    >>> stats.false_discovery_control(ps)
    array([0.0015    , 0.003     , 0.0095    , 0.035625  , 0.0603    ,
           0.06385714, 0.06385714, 0.0645    , 0.0765    , 0.486     ,
           0.58118182, 0.714875  , 0.75323077, 0.81321429, 1.        ])

    Now, the first *four* adjusted p-values fall below 0.05, so we would reject
    the null hypotheses corresponding with these *four* p-values. Rejection
    of the fourth null hypothesis was particularly important to the original
    study as it led to the conclusion that the new treatment had a
    "substantially lower in-hospital mortality rate."

    """
    # Input Validation and Special Cases
    ps = np.asarray(ps)

    ps_in_range = (np.issubdtype(ps.dtype, np.number)
                   and np.all(ps == np.clip(ps, 0, 1)))
    if not ps_in_range:
        raise ValueError("`ps` must include only numbers between 0 and 1.")

    methods = {'bh', 'by'}
    if method.lower() not in methods:
        raise ValueError(f"Unrecognized `method` '{method}'."
                         f"Method must be one of {methods}.")
    method = method.lower()

    if axis is None:
        axis = 0
        ps = ps.ravel()

    axis = np.asarray(axis)[()]
    if not np.issubdtype(axis.dtype, np.integer) or axis.size != 1:
        raise ValueError("`axis` must be an integer or `None`")

    if ps.size <= 1 or ps.shape[axis] <= 1:
        return ps[()]

    ps = np.moveaxis(ps, axis, -1)
    m = ps.shape[-1]

    # Main Algorithm
    # Equivalent to the ideas of [1] and [2], except that this adjusts the
    # p-values as described in [3]. The results are similar to those produced
    # by R's p.adjust.

    # "Let [ps] be the ordered observed p-values..."
    order = np.argsort(ps, axis=-1)
    ps = np.take_along_axis(ps, order, axis=-1)  # this copies ps

    # Equation 1 of [1] rearranged to reject when p is less than specified q
    i = np.arange(1, m+1)
    ps *= m / i

    # Theorem 1.3 of [2]
    if method == 'by':
        ps *= np.sum(1 / i)

    # accounts for rejecting all null hypotheses i for i < k, where k is
    # defined in Eq. 1 of either [1] or [2]. See [3]. Starting with the index j
    # of the second to last element, we replace element j with element j+1 if
    # the latter is smaller.
    np.minimum.accumulate(ps[..., ::-1], out=ps[..., ::-1], axis=-1)

    # Restore original order of axes and data
    np.put_along_axis(ps, order, values=ps.copy(), axis=-1)
    ps = np.moveaxis(ps, -1, axis)

    return np.clip(ps, 0, 1)
