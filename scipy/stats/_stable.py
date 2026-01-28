import numpy as np
from scipy import stats, special, integrate
from scipy.optimize import elementwise
import scipy._lib.array_api_extra as xpx
from scipy._lib._array_api import xp_promote
import matplotlib.pyplot as plt

# Stable Distribution
# [1]: Nolan, John P. "Numerical calculation of stable densities and distribution
#      functions." Communications in statistics. Stochastic models 13.4 (1997): 759-774.
# [2]: Ament, Sebastian, and Michael O’Neil. "Accurate and efficient numerical
#      calculation of stable densities via optimized quadrature and asymptotics."
#      Statistics and Computing 28.1 (2018): 171-185.

# [1], p. 761
def zeta(a, b, *, a1):
    if not a1:
        return -b * np.tan(np.pi * a / 2)
    else:
        return np.zeros_like(a)


def th0(a, b, *, a1):
    if not a1:
        return 1/a * np.atan(b * np.tan(np.pi * a / 2))  # unstable at np.pi/2
    else:
        return np.full_like(a, np.pi / 2)


def c1(a, b, *, a1):
    if not a1:
        return np.where(a < 1, 1/np.pi * (np.pi/2 - th0(a, b, a1=a1)), 1.)
    else:
        return np.zeros_like(a)


def V(th, a, b, *, a1):
    if not a1:
        th0_ = th0(a, b, a1=a1)
        t1 = (np.cos(a * th0_))**(1 / (a - 1))
        t2 = (np.cos(th) / np.sin(a * (th0_ + th)))**(a / (a - 1))
        t3 = np.cos(a*th0_ + (a - 1)*th) / np.cos(th)  # unstable at np.pi/2
        return t1 * t2 * t3
    else:
        return 2 / np.pi * (np.pi/2 + b*th) / np.cos(th) * np.exp(
            1/b * (np.pi/2 + b*th) * np.tan(th))


# # [1], p. 766, eq. (4)
def c2(x, a, b, *, a1):
    if not a1:
        return a / (np.pi * np.abs(a - 1) * (x - zeta(a, b, a1=a1)))
    else:
        return 1 / (2 * np.abs(b))


def c3(a, *, a1):
    if not a1:
        return np.sign(1 - a) / np.pi
    else:
        return np.full_like(a, 1 / np.pi)


def g(th, x, a, b, *, a1):
    if not a1:
        return (x - zeta(a, b, a1=a1))**(a/(a - 1)) * V(th, a, b, a1=a1)
    else:
        return np.exp(-np.pi/2 * x/b) * V(th, 1, b, a1=a1)


def Fi(x, a, b, *, a1):
    def integrand(th, x, a, b):
        return np.exp(-g(th, x, a, b, a1=a1))

    # When a and |b| are both ~ 1, g goes haywire at the right bound. The adjustment
    # here to the right endpoint is a numerical hack to be replaced with more stable g.
    bounds = (-th0(a, b, a1=a1), np.pi / 2 - 20*np.spacing(np.pi/2))

    integral = integrate.tanhsinh(integrand, *bounds, args=(x, a, b)).integral
    return np.asarray(c1(a, b, a1=a1) + c3(a, a1=a1) * integral)


def fi(x, a, b, *, a1):
    def integrand(th, x, a, b):
        g_ = g(th, x, a, b, a1=a1)
        res = np.asarray(g_ * np.exp(-g_))
        # [1], pp. 766-767 "is continuous, positive, strictly monotonic... Thus g has
        # the same properties..." But this is not true numerically. Experimentally,
        # when the result is negative or NaN, it should have been ~0.
        res[(res < 0) | np.isnan(res)] = 0
        return res

    bounds = (-th0(a, b, a1=a1), np.pi/2)

    # [1], p. 767. "...the integrand can be very peaked. To avoid this problem,
    # the program locates the peak (by numerically finding $\theta_2$ where g == 1...
    # The integral is then evaluated in two pieces."
    th2_res = elementwise.find_root(lambda th, x, a, b: g(th, x, a, b, a1=a1) - 1,
                                    bounds, args=(x, a, b))

    # What the paper doesn't mention:
    # In some cases, there may not be a (numerical) root in the inclusive interval,
    # so "break" the integration at an endpoint. One integral is a no-op / returns zero.
    th2 = np.where(th2_res.success, th2_res.x, np.pi/2)

    # Note: can get substantially better accuracy by increasing `minlevel`.
    integral1 = integrate.tanhsinh(integrand, bounds[0], th2, args=(x, a, b))
    integral2 = integrate.tanhsinh(integrand, th2, bounds[1], args=(x, a, b))
    integral = integral1.integral + integral2.integral

    return np.asarray(c2(x, a, b, a1=a1) * integral)


def F(x, a, b):
    x, a, b = xp_promote(x, a, b, force_floating=True, broadcast=True, xp=np)

    # [1], p. 761, Thm. 1 (c) and (d)
    a1 = a == 1
    i = (~a1 & (x >= zeta(a, b, a1=False))) | (a1 & (b > 0))
    x = np.where(i, x, -x)  # when a == 1, (d) doesn't say to negate x...
    b = np.where(i, b, -b)

    # First, assume a != 1, and just evaluate according to strategy in [1].
    res = Fi(x, a, b, a1=False)
    # a == 1 also evaluated according to [1], but the integrand takes a different form
    res[a1] = Fi(x[a1], a[a1], b[a1], a1=True)

    # Cauchy distribution
    cauchy = (a == 1) & (b == 0)
    res[cauchy] = np.atan(x[cauchy])/np.pi + 1/2

    res[~i] = 1 - res[~i]
    return res


def f(x, a, b):
    x, a, b = xp_promote(x, a, b, force_floating=True, broadcast=True, xp=np)

    # [1], p. 761, Thm. 1 (c)
    z = zeta(a, b, a1=False)
    i = x >= z
    x = np.where(i, x, -x)
    b = np.where(i, b, -b)

    # First, assume a != 1, and just evaluate according to strategy in [1].
    res = fi(x, a, b, a1=False)

    # Use asymptotic expansions for small and large |x - zeta| from [2],
    # assuming a != 1.
    z = zeta(a, b, a1=False)
    rtol = 1e-14
    eps = rtol * res
    n = 30

    # [2], eq. (2.24)
    B0 = (eps*a*np.pi*(1 + z**2)**((n + 1) / (2*a))
          * special.gamma(n + 1)/special.gamma((n + 1)/a))**(1/n)
    # "We find that truncating (2.18) after the n-1 term is accurate to within eps
    #  of the true value for all x satisfying..."
    i = np.abs(x - z) <= B0
    # [2], eq. (2.18)
    k = np.reshape(np.arange(0, n+1), (-1,) + (1,)*x.ndim)
    S0 = 1/(a*np.pi) * np.sum(special.gamma((k + 1)/a)/special.gamma(k + 1)
                              * (1 + z**2)**(-(k+1)/(2*a))
                              * np.sin((np.pi/2 + np.atan(z)/a)*(k+1)) * (x - z)**k, axis=0)
    res[i] = S0[i]

    # [2], eq. (2.29)
    B_inf = (a / (np.pi * eps) * (1 + z**2)**(n/2)
             * special.gamma(a * n) / special.gamma(n)) ** (1/(a*n - 1))
    # "... the series (2.25) is accurate to precision eps for any x satisfying..."
    i = (np.abs(x - zeta(a, b, a1=False)) > B_inf)
    # [2], eq. (2.25)
    k = np.reshape(np.arange(1, n), (-1,) + (1,)*x.ndim)
    S_inf = a / np.pi * np.sum((-1)**(k+1) * special.gamma(a*k)/special.gamma(k)
                               * (1 + z**2)**(k/2) * np.sin((np.pi*a/2 - np.atan(z))*k)
                               * (x - z)**(-a*k-1), axis=0)
    res[i] = S_inf[i]

    # a == 1 also evaluated according to [1], but the integrand takes a different form
    a1 = (a == 1)
    res[a1] = fi(x[a1], a[a1], b[a1], a1=True)

    # Cauchy distribution
    cauchy = (a == 1) & (b == 0)
    res[cauchy] = 1 / (np.pi * (1 + x[cauchy]**2))

    # Normal distribution - already accurate, so not worth the time to special case
    # normal = (a == 2)
    # res[normal] = 1/np.sqrt(4 * np.pi) * np.exp(-x[normal]**2/4)

    # Levy distribution - already accurate, and formula below adds singularity at x=0
    # levy = (a == 0.5) & (b == 1)
    # res[levy] = 1/np.sqrt(2 * np.pi) * np.exp(-1/(2*x[levy])) / x[levy]**1.5

    # Known issues:
    # Tails where a == 1, b != 0
    # a close, but not equal to, 1
    # Small a
    return res
