---
title: 'tukey_hsd: An Accurate Implementation of the Tukey Honestly Significant Difference Test in Python'
tags:
  - Python
  - SciPy
  - statistics
  - hypothesis testing
  - statistical distributions
authors:
  - name: Dominic Chmiel^[co-first author]
    affiliation: 1
  - name: Sam Wallan^[co-first author]
    affiliation: 1
  - name: Matt Haberland^[corresponding author]
    affiliation: 1
    orcid: 0000-0003-4806-3601
affiliations:
 - name: California Polytechnic State University, San Luis Obispo
   index: 1
date: 29 December 2021
bibliography: paper.bib

---

# Summary

In a world awash with data and computers, it is tempting to automate the
process of scientific discovery by performing comparisons between many pairs
of variables in hope of finding correlations. When frequentist hypothesis
tests between pairs of variables are performed at a fixed confidence level,
increasing the number of tests increases the probability of observing a
"statistically significant" result, even when the null hypothesis is actually
true. Carefully designed tests, such as Tukey's HSD (Honestly Significant
Difference) Test [@tukey1949comparing], protect against this practice of "data
dredging", producing p-values and confidence intervals that correctly account
for the number of comparisons performed. Several such tests rely on the
studentized range distribution [@lund1983algorithm], which models the range
(i.e. the difference between the maximum and minimum values) of the means of
samples from a normally distributed population. Although there are already
implementations of these tests available in the scientific Python ecosystem,
all of them rely on approximations of the studentized range distribution,
which are not be accurate outside the range of inputs for which they are
designed. Here we present the implementation of a very accurate and
sufficiently fast implementation of the studentized range distribution and a
function for performing Tukey's HSD test. Both of these are available in
SciPy 1.8.0.

# Statement of need

After Analysis of Variance (ANOVA) indicates that there is a statistically
significant difference between at least one pair of groups in an experiment,
researchers are often interested in *which* of the differences is
statistically significant. Researchers use post-hoc tests to study these
pairwise differences while controlling the experiment-wise error rate. Until
recently, no post-hoc tests were available in SciPy [@virtanen2020scipy], the
de-facto standard library of fundamental algorithms for scientific computing
in Python. To fill this gap, we contributed `scipy.stats.tukey_hsd`
[@PRtukey_hsd], a function for performing Tukey's Honestly Significant
Difference Test.

The most computationally-challenging part of implementing Tukey's HSD Test is the evaluation of the cumulative density function of the studentized range distribution, which is given by

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References