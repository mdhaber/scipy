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

`Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.

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