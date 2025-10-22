---
title: 'Vectorized, Python Array API Standard Compatible Functions for Quadrature, Series Summation, Differentiation, Optimization, and Root Finding in SciPy'
tags:
  - Python
  - SciPy
  - integration
  - differentiation
  - optimization
  - root finding
authors:
  - name: Matt Haberland^[Corresponding author]
    affiliation: 1
    orcid: 0000-0003-4806-3601
  - name: Albert Steppi
    affiliation: 2
    orcid: 0000-0001-5871-6245
  - name: Pamphile Roy
    affiliation: 3
    orcid: 0000-0001-9816-1416
  - name: Jake Bowhay
    affiliation: 4
    orcid: 0009-0002-9559-4114
affiliations:
 - name: California Polytechnic State University, San Luis Obispo, USA
   index: 1
 - name: Quansight Labs, Austin, USA
   index: 2
 - name: Consulting Manao GMBH, Vienna, Austria
   index: 3
 - name: University of Bristol, Bristol, United Kingdom
   index: 4
date: 1 March 2025
bibliography: paper.bib

---

# Summary

Numerical integration, series summation, differentiation, optimization, and root finding are fundamental problems with applications in essentially all domains of science and engineering. Frequently, such problems do not arise individually, but rather in batches; e.g., differentiation of a single curve at many points or minimization of a function for many values of a parameter. In array computing, operations on batches of values can be vectorized. With NumPy [@numpy], operations on arrays expressed in Python can be evaluated as (sequential) loops in efficient native code, or in some cases in parallel with SIMD [@nep38]. Other Python array libraries such as CuPy [@cupy], PyTorch [@pytorch], and JAX [@jax] are able to exploit GPUs to parallelize vectorized computations. However, SciPy [@scipy] – the de facto standard Python library for solving the above problems – offered few functions capable of vectorizing the solution of such problems with NumPy [@numpy], let alone with alternative array libraries.

This paper discusses several new features that fill this gap, included in SciPy 1.15.0:

- `scipy.differentiate` [@differentiate], a sub-package for numerical differentiation of scalar or vector-valued functions of one or more variables,
- `scipy.integrate.tanhsinh` [@tanhsinh] for quadrature of scalar or vector-valued integrands of one variable,
- `scipy.integrate.nsum` [@nsum] for summation of real-valued finite or infinite series,
- `scipy.optimize.elementwise` [@optimize_elementwise] for finding roots and minimization of a single-input, single-output function.

Although the details of the algorithms are inherently distinct, these features rely on a common framework for elementwise iterative methods and share similar interfaces, and the same implementation works with several Python Array API Standard [@arrayapi] compatible arrays, including CuPy and PyTorch. These features will dramatically improve performance of end-user applications, and together, they form the backbone of SciPy's new random variable infrastructure.

# Statement of need

Before the release of SciPy 1.15.0, the need for these capabilities was partially met in the scientific Python ecosystem. As popular examples, Numdifftools [@numdifftools] and PyNumDiff [@PyNumDiff] provide tools for numerical differentiation, `quadpy` [@quadpy] and `scipy.integrate` offer several functions for numerical integration, `mpmath`'s [@mpmath] `nsum` function is for series summation, and `scipy.optimize` offers functions for scalar minimization and root finding. However, to the authors' knowledge, the new implementations are unique in that they offer the following advantages.

- Backend-independence: all of these features are "Array API compatible" in the sense that they rely almost entirely on the Python standard library and calls to Python Array API Standard functions/methods. Consequently, the functions accept objects other than NumPy arrays, such as PyTorch tensors and CuPy arrays, and the calculations are performed by the underlying array library.
- Speed: the features take full advantage of vectorized user callables, avoiding slow Python loops and the excessive overhead of repeatedly calling compiled code. 
- Prevalence: SciPy is one of the most popular scientific Python packages. If a scientific Python user needs these features, chances are that they already have SciPy installed, eliminating the need to find and learn a new package.
- Ease-of-use: the API reference for these new functions is thorough, and the interfaces share common features and operate smoothly with other SciPy functions.
- Dependability: as with all new SciPy code, these features were designed and implemented following good software development practices. They have been carefully peer-reviewed, and extensive unit tests protect against backward incompatible changes and regressions.

# Acknowledgements

We gratefully acknowledge the support of Chan Zuckerberg Initiative Essential
Open Source Software for Science Grant CZI EOSS5-0000000176, "SciPy: Fundamental Tools for Biomedical Research". Thanks to all those who have submitted feature requests, bug reports, and review comments related to these features, and especially to SciPy maintainers who reviewed and merged related work.

# References