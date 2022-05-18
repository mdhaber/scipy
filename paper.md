---
title: '`tukey_hsd`: An Accurate Implementation of the Tukey Honestly
       Significant Difference Test in Python'
tags:
  - Python
  - SciPy
  - statistics
  - hypothesis testing
  - statistical distributions
authors:
  - name: Dominic Chmiel^[Co-first author]
    affiliation: 1
    orcid: 0000-0003-2499-6693
  - name: Samuel Wallan^[Co-first author]
    affiliation: 1
    orcid: 0000-0002-5952-8921
  - name: Matt Haberland^[Corresponding author]
    affiliation: 1
    orcid: 0000-0003-4806-3601
affiliations:
 - name: California Polytechnic State University, San Luis Obispo, USA
   index: 1
date: 2 February 2022
bibliography: paper.bib

---

# Summary

In a world awash with data and computers, it is tempting to automate the
process of scientific discovery by performing comparisons between many pairs
of variables in the hope of finding correlations. When frequentist hypothesis
tests are performed at a fixed confidence level,
increasing the number of tests increases the probability of observing a
"statistically significant" result, even when the null hypothesis is actually
true. Carefully-designed tests, such as Tukey's honestly significant
difference (HSD) test [@tukey1949comparing], protect against this practice of
"p-hacking" by producing p-values and confidence intervals that account
for the number of comparisons performed. Several such tests rely on the
studentized range distribution [@lund1983algorithm], which models the range
(i.e. difference between maximum and minimum values) of the means of
samples from a normally distributed population. Although there are already
implementations of these tests available in the scientific Python ecosystem,
all of them rely on approximations of the studentized range distribution,
which are not accurate outside the range of inputs for which they are
designed. Here we present the implementation of a very accurate and
sufficiently fast implementation of the studentized range distribution and a
function for performing Tukey's HSD test. A thorough assessment of the methods,
accuracy, and speed of the underlying calculations is available in
[@StudentizedRangeSciPy], and an extensive test  suite included in SciPy guards
against regressions. Both of these features are available in SciPy 1.8.0.

# Statement of need

When analysis of variance (ANOVA) indicates that there is a statistically
significant difference between at least one pair of groups in an experiment,
researchers are often interested in *which* of the differences is
statistically significant. Researchers use "post-hoc tests" to study these
pairwise differences while controlling the experiment-wise error rate. Until
recently, no post-hoc tests were available in SciPy [@virtanen2020scipy], the
de-facto standard library of fundamental algorithms for scientific computing
in Python. To fill this gap, we contributed `scipy.stats.tukey_hsd`
[@PRtukey_hsd], a function for performing Tukey's HSD test.

The most computationally-challenging part of implementing Tukey's HSD test is
the evaluation of the cumulative distribution function (CDF) of the studentized
range distribution, which is given by

\begin{eqnarray*}
F(q; k, \nu) = \frac{k\nu^{\nu/2}}{\Gamma(\nu/2)2^{\nu/2-1}}
\int_{0}^{\infty} \int_{-\infty}^{\infty} s^{\nu-1} e^{-\nu s^2/2} \phi(z)
[\Phi(sq + z) - \Phi(z)]^{k-1} \,dz \,ds
\end{eqnarray*}

where $q$ is the studentized range, $k$ is the number of groups, $\nu$ is the
number of degrees of freedom used to determine the pooled sample variance, and
$\phi(z)$ and $\Phi(z)$ represent the normal probability density function
and normal CDF, respectively.
There is no closed-form expression for this integral, and numerical
integration requires care, as naive evaluation of the integrand results
in overflow even for modest values of the parameters. Consequently, other
packages in the open-source scientific Python ecosystem, such as statsmodels
[@seabold2010statsmodels] and Pingouin [@vallat2018pingouin], have relied on
interpolation between tabulated values. To satisfy the need for a more
accurate implementation of this integral, we contributed
`scipy.stats.studentized_range` [@PRstudentized_range], a class that
evaluates the CDF and many other functions of the distribution. Both
statsmodels and Pingouin have since adopted this class to perform
studentized range distribution calculations.

# Acknowledgements

We gratefully acknowledge the support of Chan Zuckerberg Initiative Essential
Open Source Software for Science Grant EOSS-0000000432. Thanks also to
reviewers Pamphile Roy, Nicholas McKibben, and Warren Weckesser.

# References