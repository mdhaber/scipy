---
title: 'Fast Resampling and Monte Carlo Methods in Python'
tags:
  - Python
  - SciPy
  - statistics
  - bootstrap
  - confidence intervals
  - permutation tests
  - monte carlo methods
  - hypothesis testing
authors:
  - name: Matt Haberland^[Corresponding author]
    affiliation: 1
    orcid: 0000-0003-4806-3601
affiliations:
 - name: California Polytechnic State University, San Luis Obispo, USA
   index: 1
date: 3 Januarry 2023
bibliography: paper.bib

---

# Summary

In the 1935 book The Design of Experiments [@fisher1949design], Ronald Fisher analyzed the results of an experiment performed by Charles Darwin to determine the effect of cross-fertilization on plant growth [@darwin1877effects]. In these experiments, Darwin collected seeds from both cross- and self-fertilized flowers of a parent plant and germinated the seeds under identical conditions. When a pair of cross- and self-fertilized seeds happened to germinate at the same time, the two were planted in opposite sides of the same pot and nurtured in the same environment. At the end of the study, the heights of the plants were measured.

To infer whether the greater height of the cross-fertilized plants should be attributed to random chance or the change in fertilization method, Fisher computed the difference in height between paired plants and used a paired-sample t-test [@student1908probable] to estimate the statistical significance of the results. He noted that the t-test relies on assumptions about the population of height differences, and he proposed the following statistical test, which makes no such assumptions.

Under the null hypothesis that the fertilization method has no effect on plant height, the heights of the two plants in a pair are sampled from the same population. Therefore, under the null hypothesis, the sign of the difference in heights would be positive or negative with equal probability. To perform the test, Fisher considers the distribution of mean height differences under all possible combinations of signs: the "null distribution". If the observed mean height difference is greater than almost all elements in the null distribution, it is unlikely to be observed under the null hypothesis, and this is taken as evidence against the null hypothesis in favor of the alternative that cross-fertilized plants tend to be taller.

This is an early example of a *permutation test*. Like *Monte Carlo tests* [@robert1999monte] and the *bootstrap* [@efron1994introduction], permutation tests replace many assumptions and mathematical approximations with *lots* of computation. Naturally, these methods did not become very popular until the advent of the computer, but even today, some popular programming languages lack efficient implementations of these essential statistical techniques. This paper discusses the addition of functions implementing these techniques to SciPy [@virtanen2020scipy], the de facto standard library for scientific computing in Python.

# Statement of need

Scientists and engineers frequently analyze data by considering questions of the following forms.

1. Are the observations sampled from a hypothesized distribution? For instance, in Darwin's experiment, is the difference in heights between paired plants normally distributed?
2. Are the samples drawn from the *same* distribution? For example, does the fertilization method have no effect on plant height?
3. What can be inferred from the samples about the true value of a population statistic? For instance, what is the uncertainty in the true mean difference in heights between paired plants?

Monte Carlo hypothesis tests, permutation tests, and the bootstrap are general procedures for answering questions of these forms, respectively. 

Although there are a variety of specialized statistical procedures for answering such questions (e.g. normality tests, t-tests, standard error of the mean), most computer implementations provide answers that are only guaranteed to be accurate when assumptions that simplify the mathematics are met (e.g. the sample size is large, there are no ties in the data). At the expense of additional computation, the `scipy.stats` functions `monte_carlo_test`, `permutation_test`, and `bootstrap` provide accurate answers to questions 1-3 even when the simplifying assumptions of specialized procedures are not met. Furthermore, these functions can be used to answer questions 1-3 even when specialized procedures have not been *developed*, giving users tremendous flexibility in the analysis that they can perform.

Before the release of SciPy 1.9.0, the need for these procedures was partially met in the scientific Python ecosystem by tutorials (e.g. blog posts, medium.com) and niche packages, but the functions in SciPy have several advantages:

- Prevalence: SciPy is one of the most downloaded scientific Python packages. If a Python user finds the need for these statistical methods, chances are that they already have SciPy, eliminating the need to find and install a new package.
- Speed: the functions take advantage of vectorized user code, avoiding slow Python loops.
- Easy-of-use: the function API reference and tutorials are thorough, and the interfaces share common features and complement other SciPy functions.
- Dependability: as with all SciPy code, these functions were rigorously peer-reviewed, and unit tests are extensive.

# How it works

All three functions accept data and a user-defined statistic function; they differ in how these inputs are used. We conclude by describing how each function works and giving example use cases.

## Monte Carlo Tests

`scipy.stats.monte_carlo_test` uses the user-defined statistic to compare a single sample against a hypothesized statistical distribution. For example, to test Darwin's data for normality, the user passes to `monte_carlo_test` an array containing the height differences of paired plants and a statistic function sensitive to deviations from normality, such as the Anderson-Darling $A^2$ statistic [@anderson1952asymptotic] returned by `scipy.stats.anderson`. `monte_carlo_test` calculates the statistic of the observed data and the "Monte Carlo distribution", that is, the statistic of many random samples from a user-specified normal distribution. If the observed value of the statistic is extreme compared to the Monte Carlo distribution (i.e., greater than most values in the distribution, in this case), this can be taken as evidence against the null hypothesis that the data are normally distributed.

Several specialized procedures for similar statistical tests are already available, but `monte_carlo_test` is complementary. For example, `scipy.stats.jarque_bera` is a normality test that that is exact only in the limit of an infinitely large sample, whereas `monte_carlo_test` can use the statistic returned by `jarque_beta` to provide an accurate $p$-value for samples of any size. On the other hand, `scipy.stats.ks_1samp` can provide an exact $p$-value to assess whether a sample of finite size was drawn from a user-specified reference distribution, but only when the reference distribution is continuous; `monte_carlo_test` can compute an accurate $p$-value whether the reference distribution is continuous or discrete. `monte_carlo_test` can also be used to perform tests that have not already been implemented in Python, such as a version of the Anderson-Darling test for the Weibull distribution [@gh10829].

## Permutation Tests

`scipy.stats.permutation_test` uses the statistic to compare multiple data samples against one another. For example, to test Darwin's data against the null hypothesis that the fertilization method has no effect on plant height, the user passes to `permutation_test` an array containing the height differences of paired plants and a statistic sensitive to the desired effect, such as the sample mean. `permutation_test` calculates the mean of the observed data and the "null distribution" of the mean under each possible combination of sign changes of the observations. If the observed value of the statistic is extreme compared to the null distribution, this can be taken as evidence against the null hypothesis that the observed difference in sample means is due to chance.

Note that `permutation_test` can also be configured to perform other types of "permutations" (rearrangements of the data). For example, suppose that the observations in the self-fertilized and cross-fertilized samples were not paired. In this case, an appropriate statistic would be the difference between sample means (rather than the mean of paired differences), and the null distribution would be assembled by calculating the statistic for all possible partitions of the observations into two samples of the original sizes rather than permuting signs. Such a test would be akin the *independent* sample t-test. A permutation type appropriate for correlation tests is also available, and all permutation types are generalized to support an arbitrary number of samples.

The number of possible permutations grows very quickly with sample size. When this number exceeds a user-defined maximum, `permutation_test` performs the maximum number of *random* permutations to approximate the null distribution, using a biased estimate of the $p$-value to control the type I error rate [@phipson2010permutation].

Several specialized procedures for similar statistical tests are already available, but `permutation_test` is complementary. For instance, `scipy.stats.spearmanr` is a test of correlation that reports a $p$-value that is exact only in the limit of infinitely large samples, whereas `permutation_test` can use the statistic returned by `spearmanr` to provide an accurate $p$-value for small samples. On the other hand, `scipy.stats.wilcoxon` can provide an exact $p$-value to assess whether paired samples are drawn from the same distribution, but only when there are no ties in the data; `permutation_test` has no trouble producing exact $p$-values in the presence of ties. `permutation_test` can also be used to perform tests that have not already been implemented in Python, such as a weighted two-sample Kolmogorov-Smirnov test [@gh12315].

## The Bootstrap

`scipy.stats.bootstrap` uses the data and sample statistic to make inferences about a corresponding statistic of the population(s) from which the data were drawn. For example, to estimate a confidence interval on the true mean of paired plant height differences, the user passes to `bootstrap` an array containing the height differences of paired plants and a function that computes the sample mean of paired plant height differences. `bootstrap` calculates the statistic of the observed data and computes a "bootstrap distribution" of the statistic, that is, the statistic value for many artificial samples generated by randomly resampling with replacement from the observed data. The simplest way `bootstrap` can approximate a 95% confidence interval is to return the 2.5 and 97.5 percentile values from the bootstrap distribution, but a more accurate method that corrects for bias and skewness is also available.

Besides the bootstrap, a few specialized procedures for generating confidence intervals and standard errors have been developed, but they often rely on normality approximations and, with few exceptions, they are exact only in the limit of infinitely large samples. `bootstrap` computes confidence intervals and standard errors of most user-defined statistics under a comparatively mild assumption that the data are representative of the underlying distribution(s).

# Acknowledgements

We gratefully acknowledge the support of Chan Zuckerberg Initiative Essential
Open Source Software for Science Grant EOSS-0000000432 and the 2022 Round 1 NumFOCUS Small Development Grant "Introducing Users to Powerful New Features of SciPy". Thanks to all those who have submitted feature requests, bug reports, and review comments related to these features, and especially to SciPy maintainers Christoph Baumgarten, Robert Lucas, Tirth Patel, Pamphile Roy, and Warren Weckesser who reviewed this work.

# References