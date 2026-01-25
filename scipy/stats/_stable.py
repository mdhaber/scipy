import numpy as np
from scipy import stats, special, integrate
from scipy.optimize import elementwise
import scipy._lib.array_api_extra as xpx
from scipy._lib._array_api import xp_promote
import matplotlib.pyplot as plt


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

    integral = integrate.tanhsinh(integrand, -th0(a, b, a1=a1), np.pi / 2, args=(x, a, b)).integral
    return c1(a, b, a1=a1) + c3(a, a1=a1) * integral


def fi(x, a, b, *, a1):
    def integrand(th, x, a, b):
        g_ = g(th, x, a, b, a1=a1)
        res = g_ * np.exp(-g_)
        res[~(res >= 0)] = 0  # small effect at a == 1
        return res
        # return g_ * np.exp(-g_)

    # re-evaluate nextafter on the right
    bounds = (-th0(a, b, a1=a1), np.pi/2)

    # This seems to be more trouble than it's worth - root finding doesn't work reliably
    # when |b|=1 or a=2 when the roots are theoretically at interval endpoints
    th2 = elementwise.find_root(lambda th, x, a, b: g(th, x, a, b, a1=a1) - 1,
                                bounds, args=(x, a, b))
    # th2 = elementwise.find_minimum(lambda th, x, a, b: -integrand(th, x, a, b),
    #                                (bounds[0], (bounds[0]+bounds[1])/1, bounds[1]), args=(x, a, b))
    integral1 = integrate.tanhsinh(integrand, bounds[0], th2.x, args=(x, a, b), minlevel=4, rtol=1e-12)
    integral2 = integrate.tanhsinh(integrand, th2.x, np.pi/2, args=(x, a, b), rtol=1e-12)
    integral = integral1.integral + integral2.integral

    # integral = integrate.tanhsinh(integrand, *bounds, args=(x, a, b)).integral
    return c2(x, a, b, a1=a1) * integral


def F(x, a, b):
    x, a, b = xp_promote(x, a, b, force_floating=True, broadcast=True, xp=np)
    a1 = np.all(a == 1)
    i = (b > 0) if a1 else (x >= zeta(a, b, a1=a1))
    # ideally, a lazy-select function would let us handle a=1, b=0 (Cauchy)
    return xpx.apply_where(i, (x, a, b),
                           lambda x, a, b: Fi(x, a, b, a1=a1),
                           lambda x, a, b: 1 - Fi(-x, a, -b, a1=a1))[()]


def f(x, a, b):
    x, a, b = xp_promote(x, a, b, force_floating=True, broadcast=True, xp=np)

    z = zeta(a, b, a1=False)
    i = x >= z
    x = np.where(i, x, -x)
    b = np.where(i, b, -b)

    # Nolan
    res = fi(x, a, b, a1=False)

    # Expansion for |x - zeta| small
    z = zeta(a, b, a1=False)
    rtol = 1e-14
    eps = rtol * res
    n = 30
    B0 = (eps*a*np.pi*(1 + z**2)**((n + 1) / (2*a))
          * special.gamma(n + 1)/special.gamma((n + 1)/a))**(1/n)
    k = np.arange(0, n+1)
    k = np.reshape(k, (-1,) + (1,)*x.ndim)
    i = np.abs(x - z) <= B0
    S0 = 1/(a*np.pi) * np.sum(special.gamma((k + 1)/a)/special.gamma(k + 1)
                              * (1 + z**2)**(-(k+1)/(2*a))
                              * np.sin((np.pi/2 + np.atan(z)/a)*(k+1)) * (x - z)**k, axis=0)
    res[i] = S0[i]

    # Expansion for |x - zeta| large
    rtol = 1e-14
    eps = rtol * res
    n = 30
    B_inf = (a / (np.pi * eps) * (1 + z**2)**(n/2)
             * special.gamma(a * n) / special.gamma(n)) ** (1/(a*n - 1))
    i = (np.abs(x - zeta(a, b, a1=False)) > B_inf)
    k = np.arange(1, n)
    k = np.reshape(k, (-1,) + (1,)*x.ndim)
    S_inf = a / np.pi * np.sum((-1)**(k+1) * special.gamma(a*k)/special.gamma(k)
                               * (1 + z**2)**(k/2) * np.sin((np.pi*a/2 - np.atan(z))*k)
                               * (x - z)**(-a*k-1), axis=0)
    res[i] = S_inf[i]

    # a == 1 (all formulas special-cased)
    a1 = (a == 1)
    res[a1] = fi(x[a1], a[a1], b[a1], a1=True)

    # Normal
    a2 = (a == 2)
    res[a2] = 1/np.sqrt(4 * np.pi) * np.exp(-x[a2]**2/4)

    # Cauchy
    a1b0 = (a == 1) & (b == 0)
    res[a1b0] = 1 / (np.pi * (1 + x[a1b0]**2))

    return res
