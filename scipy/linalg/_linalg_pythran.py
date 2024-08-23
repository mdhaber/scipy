#pythran export _funm_loops(float32[:, :], float32[:, :], int, float)
#pythran export _funm_loops(float64[:, :], float64[:, :], int, float)
#pythran export _funm_loops(complex64[:, :], complex64[:, :], int, float)
#pythran export _funm_loops(complex128[:, :], complex128[:, :], int, float)
def _funm_loops(F, T, n, minden):
    for p in range(1, n):
        for i in range(1, n - p + 1):
            j = i + p
            s = T[i - 1, j - 1] * (F[j - 1, j - 1] - F[i - 1, i - 1])
            ks = slice(i, j - 1)
            val = sum(T[i - 1, ks] * F[ks, j - 1]) - sum(F[i - 1, ks] * T[ks, j - 1])
            s = s + val
            den = T[j - 1, j - 1] - T[i - 1, i - 1]
            if den != 0.0:
                s = s / den
            F[i - 1, j - 1] = s
            minden = min(minden, abs(den))
    return F, minden
