import numpy as np
from scipy import stats, special, integrate
from scipy.optimize import elementwise
import scipy._lib.array_api_extra as xpx
from scipy._lib._array_api import xp_promote


def zeta(a, b):
    return -b * np.tan(np.pi * a / 2)


def zetaa1(a, b):
    return np.zeros_like(a)


def th0(a, b):
    return 1 / a * np.atan(b * np.tan(np.pi * a / 2))


def th0a1(a, b):
    return np.full_like(a, np.pi/2)


def c1(a, b):
    return np.where(a < 1, 1/np.pi * (np.pi/2 - th0(a, b)), 1.)


def c1a1(a, b):
    return np.zeros_like(a)


def V(th, a, b):
    th0_ = th0(a, b)
    t1 = (np.cos(a * th0_))**(1 / (a - 1))
    t2 = (np.cos(th) / np.sin(a * (th0_ + th)))**(a / (a - 1))
    t3 = np.cos(a*th0_ + (a - 1)*th) / np.cos(th)
    return t1 * t2 * t3


def Va1(th, a, b):
    return 2/np.pi * (np.pi/2 + b*th) / np.cos(th) * np.exp(
        1/b * (np.pi/2 + b*th) * np.tan(th))


def c2(x, a, b):
    return a / (np.pi * np.abs(a - 1) * (x - zeta(a, b)))


def c2a1(x, a, b):
    return 1 / (2 * np.abs(b))


def c3(a):
    return np.sign(1 - a) / np.pi


def c3a1(a):
    return np.full_like(a, 1 / np.pi)


def g(th, x, a, b):
    return (x - zeta(a, b)) ** (a / (a - 1)) * V(th, a, b)


def ga1(th, x, a, b):
    return np.exp(-np.pi / 2 * x / b) * Va1(th, 1, b)


def Fi(x, a, b, *, a1):
    g_fun, th0_fun, c1_fun, c3_fun = (ga1, th0a1, c1a1, c3a1) if a1 else (g, th0, c1, c3)

    def integrand(th, x, a, b):
        return np.exp(-g_fun(th, x, a, b))

    integral = integrate.tanhsinh(integrand, -th0_fun(a, b), np.pi / 2, args=(x, a, b)).integral
    return c1_fun(a, b) + c3_fun(a) * integral


def fi(x, a, b, *, a1):
    g_fun, th0_fun, c2_fun = (ga1, th0a1, c2a1) if a1 else (g, th0, c2)

    def integrand(th, x, a, b):
        g_ = g_fun(th, x, a, b)
        return g_ * np.exp(-g_)

    bounds = (np.nextafter(-th0_fun(a, b), np.inf), np.nextafter(np.pi / 2, -np.inf))

    # root-finding doesn't work in the left tail (|b| ~ 1)
    # th2 = elementwise.find_root(lambda th, x, a, b: g(th, x, a, b) - 1,
    #                             bounds, args=(x, a, b))
    # integral1 = integrate.tanhsinh(integrand, bounds[0], th2.x, args=(x, a, b))
    # integral2 = integrate.tanhsinh(integrand, th2.x, np.pi/2, args=(x, a, b))
    # integral = integral1.integral + integral2.integral

    integral = integrate.tanhsinh(integrand, *bounds, args=(x, a, b)).integral
    return c2_fun(x, a, b) * integral


def F(x, a, b):
    x, a, b = xp_promote(x, a, b, force_floating=True, broadcast=True, xp=np)
    a1 = np.all(a == 1)
    i = (b > 0) if a1 else (x >= zeta(a, b))
    return xpx.apply_where(i, (x, a, b),
                           lambda x, a, b: Fi(x, a, b, a1=a1),
                           lambda x, a, b: 1 - Fi(-x, a, -b, a1=a1))[()]


def f(x, a, b):
    x, a, b = xp_promote(x, a, b, force_floating=True, broadcast=True, xp=np)
    a1 = np.all(a == 1)
    i = (x >= zetaa1(a, b)) if a1 else (x >= zeta(a, b))
    return xpx.apply_where(i, (x, a, b),
                           lambda x, a, b: fi(x, a, b, a1=a1),
                           lambda x, a, b: fi(-x, a, -b, a1=a1))[()]
