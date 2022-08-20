from functools import cached_property

import numpy as np
from scipy import linalg
from . import _multivariate

__all__ = ["Covariance", "CovViaDiagonal", "CovViaPrecision",
           "CovViaEigendecomposition", "CovViaCov", "CovViaPSD"]


class Covariance():
    """
    Representation of a covariance matrix as needed by multivariate_normal
    """

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
        if m != n or A.ndim != 2 or not np.issubdtype(A.dtype, np.number):
            message = (f"The input `{name}` must be a square, "
                       "two-dimensional array of numbers.")
            raise ValueError(message)
        return A

    def _validate_vector(self, A, name):
        A = np.atleast_1d(A)
        if A.ndim != 1 or not np.issubdtype(A.dtype, np.number):
            message = (f"The input `{name}` must be a one-dimensional array "
                       "of numbers.")
            raise ValueError(message)
        return A


class CovViaDiagonal(Covariance):
    """
    Representation of a covariance provided via the diagonal
    """

    def __init__(self, diagonal):
        diagonal = self._validate_vector(diagonal, 'diagonal')

        i_positive = diagonal > 0
        positive_diagonal = diagonal[i_positive]
        self._log_pdet = np.sum(np.log(positive_diagonal))

        psuedo_reciprocals = np.zeros_like(diagonal, dtype=np.float64)
        psuedo_reciprocals[i_positive] = 1 / np.sqrt(positive_diagonal)

        self._LP = psuedo_reciprocals
        self._rank = positive_diagonal.shape[-1]
        self._A = np.diag(diagonal)
        self._dimensionality = diagonal.shape[-1]
        self._i_positive = i_positive
        self._allow_singular = True

    def _whiten(self, x):
        return x * self._LP

    def _support_mask(self, x):
        """
        Check whether x lies in the support of the distribution.
        """
        return np.all(x[..., ~self._i_positive] == 0, axis=-1)


class CovViaPrecision(Covariance):
    """
    Representation of a covariance provided via the precision matrix
    """

    def __init__(self, precision, covariance=None):
        precision = self._validate_matrix(precision, 'precision')
        if covariance is not None:
            covariance = self._validate_matrix(precision, 'covariance')

        self._LP = np.linalg.cholesky(precision)
        self._log_pdet = -2*np.log(np.diag(self._LP)).sum(axis=-1)
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

        self._factor = np.linalg.cholesky(cov.T[::-1, ::-1])[::-1, ::-1]
        self._log_pdet = 2*np.log(np.diag(self._factor)).sum(axis=-1)
        self._rank = cov.shape[-1]  # must be full rank for cholesky
        self._A = cov
        self._dimensionality = self._rank
        self._allow_singular = False

    def _whiten(self, x):
        return linalg.solve_triangular(self._factor, x.T, lower=False).T


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
