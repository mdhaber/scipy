import numpy as np
import timeit
from numpy.testing import assert_allclose, assert_equal
from scipy import stats, optimize, integrate, linalg
import matplotlib.pyplot as plt
from mpmath import mp
mp.dps = 50
rng = np.random.default_rng(1638083107694713882823079058616272161)
from scipy.linalg._linalg_pythran import levinson as levinson_pythran
from scipy.linalg._solve_toeplitz import levinson

ns = np.logspace(1, 3)

t_pythran = []
t_cython = []

for n in ns:
    n = int(n)
    print(n)
    x = rng.random(2*n-1)
    x[n-1] = np.sum(x)
    c, r = x[n-1::-1], x[n-1:]
    A = linalg.toeplitz(c, r)
    b = rng.random(n)
    # ref = linalg.solve(A, b)
    ref2 = linalg.solve_toeplitz((c, r), b)
    x = x[::-1].copy()

    res = levinson_pythran(x, b)
    res2 = levinson(x, b)
    np.testing.assert_allclose(res[0], res2[0], rtol=1e-13, atol=1e-16)
    np.testing.assert_allclose(res[1], res2[1], rtol=1e-13, atol=1e-16)

    t_pythran.append(timeit.timeit(lambda: levinson_pythran(x, b), number=500))
    t_cython.append(timeit.timeit(lambda: levinson(x, b), number=500))

plt.loglog(ns, t_pythran, label='pythran')
plt.loglog(ns, t_cython, label='cython')
plt.xlabel('Problem size $n$')
plt.ylabel('Execution time (s)')
plt.legend()
plt.show()
