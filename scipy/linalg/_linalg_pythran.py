import numpy as np

# pythran export _cholesky_update(float32[:, :], float32[:], float, bool)
# pythran export _cholesky_update(float64[:, :], float64[:], float, bool)
def _cholesky_update(R, z, eps, downdate):
    # Initialization
    alpha, beta = np.empty_like(z), np.empty_like(z)
    alpha[-1], beta[-1] = 1., 1.
    sign = -1 if downdate else 1

    # Main algorithm
    n = R.shape[0]
    for r in range(n):
        a = z[r] / R[r, r]
        alpha[r] = alpha[r - 1] + sign * a**2
        if alpha[r] < eps:  # numerically zero or negative
            return None
        beta[r] = alpha[r]**0.5
        z[r + 1:] -= a * R[r, r + 1:]
        R[r, r:] *= beta[r] / beta[r - 1]
        R[r, r + 1:] += sign * a / (beta[r] * beta[r - 1]) * z[r + 1:]

    return R
