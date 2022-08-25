from functools import cached_property

import numpy as np
from scipy import linalg
from . import _multivariate

__all__ = ["Covariance", "CovViaDiagonal", "CovViaPrecision",
           "CovViaEigendecomposition", "CovViaCov", "CovViaPSD"]


def _apply_over_matrices(f):
    def _f_wrapper(x, *args, **kwargs):
        x = np.asarray(x)
        if x.ndim <= 2:
            return f(x, *args, **kwargs)

        old_shape = list(x.shape)
        m, n = old_shape[-2:]
        new_shape = old_shape[:-2] + [m*n]

        def _f_on_raveled(x):
            return f(x.reshape((m, n)), *args, **kwargs)

        return np.apply_along_axis(_f_on_raveled, axis=-1, arr=x.reshape(new_shape))
    return _f_wrapper


@_apply_over_matrices
def _extract_diag(A):
    return np.diag(A)


@_apply_over_matrices
def _solve_triangular(A, *args, **kwargs):
    return linalg.solve_triangular(A, *args, **kwargs)


def _T(A):
    return np.swapaxes(A, -1, -2)


def _J(A):
    return np.flip(A, axis=(-1, -2))


class Covariance():
    """
    Representation of a covariance matrix as needed by multivariate_normal
    """
    # The last two axes of matrix-like input represent the dimensionality
    # In the case of diagonal elements or eigenvalues, in which a diagonal
    # matrix has been reduced to one dimension, the last dimension
    # of the input represents the dimensionality. Internally, it
    # will be expanded in the second to last axis as needed.
    # Matrix math works, but instead of the fundamental matrix-vector
    # operation being A@x, think of x are row vectors that pre-multiply A.

    def whiten(self, x):
        """
        Right multiplication by the left square root of the precision matrix.
        """
        return self._whiten(x)

    @cached_property
    def log_pdet(self):
        """
        Log of the pseudo-determinant of the covariance matrix
        """
        return self._log_pdet

    @cached_property
    def rank(self):
        """
        Rank of the covariance matrix
        """
        return self._rank

    @cached_property
    def A(self):
        """
        Explicit representation of the covariance matrix
        """
        return self._A

    @cached_property
    def dimensionality(self):
        """
        Dimensionality of the vector space
        """
        return self._dimensionality

    def _validate_matrix(self, A, name):
        A = np.atleast_2d(A)
        m, n = A.shape[-2:]
        if m != n or not np.issubdtype(A.dtype, np.number):
            message = (f"`{name}` must be an array of numbers, and "
                       f"`{name}.shape[-2]` must equal `{name}.shape[-1]`")
            raise ValueError(message)
        return A

    def _validate_vector(self, A, name):
        A = np.atleast_1d(A)
        if not np.issubdtype(A.dtype, np.number):
            message = (f"The input `{name}` must be an array of numbers.")
            raise ValueError(message)
        return A


class CovViaDiagonal(Covariance):
    """
    Representation of a covariance provided via the diagonal
    """

    def __init__(self, diagonal):
        diagonal = self._validate_vector(diagonal, 'diagonal')

        i_zero = diagonal <= 0
        positive_diagonal = np.array(diagonal, dtype=np.float64)

        positive_diagonal[i_zero] = 1  # ones don't affect determinant
        self._log_pdet = np.sum(np.log(positive_diagonal), axis=-1)

        psuedo_reciprocals = 1 / np.sqrt(positive_diagonal)
        psuedo_reciprocals[i_zero] = 0

        self._LP = np.expand_dims(psuedo_reciprocals, -2)
        self._rank = positive_diagonal.shape[-1] - i_zero.sum(axis=-1)
        self._A = np.apply_along_axis(np.diag, -1, diagonal)
        self._dimensionality = diagonal.shape[-1]
        self._i_zero = np.expand_dims(i_zero, -2)
        self._allow_singular = True

    def _whiten(self, x):
        return x * self._LP

    def _support_mask(self, x):
        """
        Check whether x lies in the support of the distribution.
        """
        return ~np.any(x * self._i_zero, axis=-1)


class CovViaPrecision(Covariance):
    """
    Representation of a covariance provided via the precision matrix
    """

    def __init__(self, precision, covariance=None):
        precision = self._validate_matrix(precision, 'precision')
        if covariance is not None:
            covariance = self._validate_matrix(precision, 'covariance')

        self._LP = np.linalg.cholesky(precision)
        self._log_pdet = -2*np.log(_extract_diag(self._LP)).sum(axis=-1)
        self._rank = precision.shape[-1]  # must be full rank in invertible
        self._precision = precision
        self._covariance = covariance
        self._dimensionality = self._rank
        self._allow_singular = False

    def _whiten(self, x):
        return x @ self._LP

    @cached_property
    def _A(self):
        return (np.linalg.inv(self._precision) if self._covariance is None
                else self._covariance)


class CovViaEigendecomposition(Covariance):
    """
    Representation of a covariance provided via eigenvalues and eigenvectors
    """

    def __init__(self, eigendecomposition):
        eigenvalues, eigenvectors = eigendecomposition
        eigenvalues = self._validate_vector(eigenvalues, 'eigenvalues')
        eigenvectors = self._validate_matrix(eigenvectors, 'eigenvectors')

        i_positive = eigenvalues > 0
        positive_eigenvalues = eigenvalues[i_positive]
        self._log_pdet = np.sum(np.log(positive_eigenvalues))

        psuedo_reciprocals = np.zeros_like(eigenvalues)
        psuedo_reciprocals[i_positive] = 1 / np.sqrt(positive_eigenvalues)

        self._LP = eigenvectors * psuedo_reciprocals
        self._rank = positive_eigenvalues.shape[-1]
        self._w = eigenvalues
        self._v = eigenvectors
        self._dimensionality = eigenvalues.shape[-1]
        self._null_basis = eigenvectors[..., ~i_positive]
        self._eps = _multivariate._eigvalsh_to_eps(eigenvalues) * 10**3
        self._allow_singular = True

    def _whiten(self, x):
        return x @ self._LP

    @cached_property
    def _A(self):
        return (self._v * self._w) @ self._v.swapaxes(-2, -1)

    def _support_mask(self, x):
        """
        Check whether x lies in the support of the distribution.
        """
        residual = np.linalg.norm(x @ self._null_basis, axis=-1)
        in_support = residual < self._eps
        return in_support


class CovViaCov(Covariance):
    """
    Representation of a covariance provided via the precision matrix
    """

    def __init__(self, cov):
        cov = self._validate_matrix(cov, 'cov')

        self._factor = _J(np.linalg.cholesky(_J(_T(cov))))
        self._log_pdet = 2*np.log(_extract_diag(self._factor)).sum(axis=-1)
        self._rank = cov.shape[-1]  # must be full rank for cholesky
        self._A = cov
        self._dimensionality = self._rank
        self._allow_singular = False

    def _whiten(self, x):
        return _T(_solve_triangular(self._factor, _T(x), lower=False))


class CovViaPSD(Covariance):
    """
    Representation of a covariance provided via an instance of _PSD
    """

    def __init__(self, psd):
        self._LP = psd.U
        self._log_pdet = psd.log_pdet
        self._rank = psd.rank
        self._A = psd._M
        self._dimensionality = psd._M.shape[-1]
        self._psd = psd
        self._allow_singular = False  # by default

    def _whiten(self, x):
        return x @ self._LP

    def _support_mask(self, x):
        return self._psd._support_mask(x)
