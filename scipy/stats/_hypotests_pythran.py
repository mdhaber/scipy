from math import factorial
import numpy as np

#pythran export _Aij(float[:,:], int, int)
#pythran export _Aij(int[:,:], int, int)
def _Aij(A, i, j):
    """Sum of upper-left and lower right blocks of contingency table."""
    # See `somersd` References [2] bottom of page 309
    return A[:i, :j].sum() + A[i+1:, j+1:].sum()

#pythran export _Dij(float[:,:], int, int)
#pythran export _Dij(int[:,:], int, int)
def _Dij(A, i, j):
    """Sum of lower-left and upper-right blocks of contingency table."""
    # See `somersd` References [2] bottom of page 309
    return A[i+1:, :j].sum() + A[:i, j+1:].sum()

#pythran export _P(float[:,:])
#pythran export _P(int[:,:])
def _P(A):
    """Twice the number of concordant pairs, excluding ties."""
    # See `somersd` References [2] bottom of page 309
    m, n = A.shape
    count = 0
    for i in range(m):
        for j in range(n):
            count += A[i, j]*_Aij(A, i, j)
    return count

#pythran export _Q(float[:,:])
#pythran export _Q(int[:,:])
def _Q(A):
    """Twice the number of discordant pairs, excluding ties."""
    # See `somersd` References [2] bottom of page 309
    m, n = A.shape
    count = 0
    for i in range(m):
        for j in range(n):
            count += A[i, j]*_Dij(A, i, j)
    return count

#pythran export _a_ij_Aij_Dij2(float[:,:])
#pythran export _a_ij_Aij_Dij2(int[:,:])
def _a_ij_Aij_Dij2(A):
    """A term that appears in the ASE of Kendall's tau and Somers' D."""
    # See `somersd` References [2] section 4: Modified ASEs to test the null hypothesis...
    m, n = A.shape
    count = 0
    for i in range(m):
        for j in range(n):
            count += A[i, j]*(_Aij(A, i, j) - _Dij(A, i, j))**2
    return count

#pythran export _comb(int, int)
def _comb(n, k):

    m = int(factorial(n)/(factorial(n-k)*factorial(k)))
    combs = np.empty((m, k))
    j = 0

    b = np.arange(k)
    combs[0] = b
    j += 1

    maxs = b + n - k
    i = k-1
    while i >= 0:
        if b[i] < maxs[i]:
            b[i:] = np.arange(b[i] + 1, b[i] + k - i + 1)
            i = k - 1
            combs[j] = b
            j += 1
        else:
            i -= 1

    return combs

#pythran export _comb2(int, int, int[:], int)
def _comb2(n, k, b, batch):
    b = b.copy()
    combs = np.empty((batch, k), dtype=int)

    maxs = np.arange(k) + n - k
    j = 0
    i = k-1
    while i >= 0 and j < batch:
        if b[i] < maxs[i]:
            b[i:] = np.arange(b[i] + 1, b[i] + k - i + 1, dtype=int)
            i = k - 1
            combs[j] = b
            j += 1
        else:
            i -= 1

    return combs
