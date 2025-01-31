import numpy as np
from scipy.special import ndtr as phid


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


# pythran export _concordant_pairs(float[:,:])
# pythran export _concordant_pairs(int[:,:])
def _concordant_pairs(A):
    """Twice the number of concordant pairs, excluding ties."""
    # See `somersd` References [2] bottom of page 309
    m, n = A.shape
    count = 0
    for i in range(m):
        for j in range(n):
            count += A[i, j]*_Aij(A, i, j)
    return count


# pythran export _discordant_pairs(float[:,:])
# pythran export _discordant_pairs(int[:,:])
def _discordant_pairs(A):
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
    # See `somersd` References [2] section 4:
    # Modified ASEs to test the null hypothesis...
    m, n = A.shape
    count = 0
    for i in range(m):
        for j in range(n):
            count += A[i, j]*(_Aij(A, i, j) - _Dij(A, i, j))**2
    return count


#pythran export _compute_outer_prob_inside_method(int64, int64, int64, int64)
def _compute_outer_prob_inside_method(m, n, g, h):
    """
    Count the proportion of paths that do not stay strictly inside two
    diagonal lines.

    Parameters
    ----------
    m : integer
        m > 0
    n : integer
        n > 0
    g : integer
        g is greatest common divisor of m and n
    h : integer
        0 <= h <= lcm(m,n)

    Returns
    -------
    p : float
        The proportion of paths that do not stay inside the two lines.

    The classical algorithm counts the integer lattice paths from (0, 0)
    to (m, n) which satisfy |x/m - y/n| < h / lcm(m, n).
    The paths make steps of size +1 in either positive x or positive y
    directions.
    We are, however, interested in 1 - proportion to computes p-values,
    so we change the recursion to compute 1 - p directly while staying
    within the "inside method" a described by Hodges.

    We generally follow Hodges' treatment of Drion/Gnedenko/Korolyuk.
    Hodges, J.L. Jr.,
    "The Significance Probability of the Smirnov Two-Sample Test,"
    Arkiv fiur Matematik, 3, No. 43 (1958), 469-86.

    For the recursion for 1-p see
    Viehmann, T.: "Numerically more stable computation of the p-values
    for the two-sample Kolmogorov-Smirnov test," arXiv: 2102.08037

    """
    # Probability is symmetrical in m, n.  Computation below uses m >= n.
    if m < n:
        m, n = n, m
    mg = m // g
    ng = n // g

    # Count the integer lattice paths from (0, 0) to (m, n) which satisfy
    # |nx/g - my/g| < h.
    # Compute matrix A such that:
    #  A(x, 0) = A(0, y) = 1
    #  A(x, y) = A(x, y-1) + A(x-1, y), for x,y>=1, except that
    #  A(x, y) = 0 if |x/m - y/n|>= h
    # Probability is A(m, n)/binom(m+n, n)
    # Optimizations exist for m==n, m==n*p.
    # Only need to preserve a single column of A, and only a
    # sliding window of it.
    # minj keeps track of the slide.
    minj, maxj = 0, min(int(np.ceil(h / mg)), n + 1)
    curlen = maxj - minj
    # Make a vector long enough to hold maximum window needed.
    lenA = min(2 * maxj + 2, n + 1)
    # This is an integer calculation, but the entries are essentially
    # binomial coefficients, hence grow quickly.
    # Scaling after each column is computed avoids dividing by a
    # large binomial coefficient at the end, but is not sufficient to avoid
    # the large dynamic range which appears during the calculation.
    # Instead we rescale based on the magnitude of the right most term in
    # the column and keep track of an exponent separately and apply
    # it at the end of the calculation.  Similarly when multiplying by
    # the binomial coefficient
    dtype = np.float64
    A = np.ones(lenA, dtype=dtype)
    # Initialize the first column
    A[minj:maxj] = 0.0
    for i in range(1, m + 1):
        # Generate the next column.
        # First calculate the sliding window
        lastminj, lastlen = minj, curlen
        minj = max(int(np.floor((ng * i - h) / mg)) + 1, 0)
        minj = min(minj, n)
        maxj = min(int(np.ceil((ng * i + h) / mg)), n + 1)
        if maxj <= minj:
            return 1.0
        # Now fill in the values. We cannot use cumsum, unfortunately.
        val = 0.0 if minj == 0 else 1.0
        for jj in range(maxj - minj):
            j = jj + minj
            val = (A[jj + minj - lastminj] * i + val * j) / (i + j)
            A[jj] = val
        curlen = maxj - minj
        if lastlen > curlen:
            # Set some carried-over elements to 1
            A[maxj - minj:maxj - minj + (lastlen - curlen)] = 1

    return A[maxj - minj - 1]


# pythran export siegelslopes(float32[:], float32[:], str)
# pythran export siegelslopes(float64[:], float64[:], str)
def siegelslopes(y, x, method):
    deltax = np.expand_dims(x, 1) - x
    deltay = np.expand_dims(y, 1) - y
    slopes, intercepts = [], []

    for j in range(len(x)):
        id_nonzero, = np.nonzero(deltax[j, :])
        slopes_j = deltay[j, id_nonzero] / deltax[j, id_nonzero]
        medslope_j = np.median(slopes_j)
        slopes.append(medslope_j)
        if method == 'separate':
            z = y*x[j] - y[j]*x
            medintercept_j = np.median(z[id_nonzero] / deltax[j, id_nonzero])
            intercepts.append(medintercept_j)

    medslope = np.median(np.asarray(slopes))
    if method == "separate":
        medinter = np.median(np.asarray(intercepts))
    else:
        medinter = np.median(y - medslope*x)

    return medslope, medinter


# pythran export _poisson_binom_pmf(float64[:])
def _poisson_binom_pmf(p):
    # implemented from poisson_binom [2] Equation 2
    n = p.shape[0]
    pmf = np.zeros(n + 1, dtype=np.float64)
    pmf[:2] = 1 - p[0], p[0]
    for i in range(1, n):
        tmp = pmf[:i+1] * p[i]
        pmf[:i+1] *= (1 - p[i])
        pmf[1:i+2] += tmp
    return pmf


# pythran export _poisson_binom(int64[:], float64[:, :], str)
def _poisson_binom(k, args, tp):
    # PDF/CDF of Poisson binomial distribution
    # k - arguments, shape (m,)
    # args - shape parameters, shape (n, m)
    # kind - {'pdf', 'cdf'}
    n, m = args.shape  # number of shapes, batch size
    cache = {}
    out = np.zeros(m, dtype=np.float64)
    for i in range(m):
        p = tuple(args[:, i])
        if p not in cache:
            pmf = _poisson_binom_pmf(args[:, i])
            cache[p] = np.cumsum(pmf) if tp=='cdf' else pmf
        out[i] = cache[p][k[i]]
    return out


### Bivariate Normal
#BVNU
#  A function for computing bivariate normal probabilities.
#  bvnu calculates the probability that x > dh and y > dk.
#    parameters
#      dh 1st lower integration limit
#      dk 2nd lower integration limit
#      r   correlation coefficient
#  Example: p = bvnu( -3, -1, .35 )
#  Note: to compute the probability that x < dh and y < dk,
#        use bvnu( -dh, -dk, r ).
#
#BVN
#  A function for computing bivariate normal probabilities.
#  bvn calculates the probability that
#    xl < x < xu and yl < y < yu,
#  with correlation coefficient r.
#   p = bvn( xl, xu, yl, yu, r )
#
#   Author
#       Alan Genz
#       Department of Mathematics
#       Washington State University
#       Pullman, Wa 99164-3113
#       Email : alangenz@wsu.edu
#
#    This function is based on the method described by 
#        Drezner, Z and G.O. Wesolowsky, (1989),
#        On the computation of the bivariate normal inegral,
#        Journal of Statist. Comput. Simul. 35, pp. 101-107,
#    with major modifications for double precision, for |r| close to 1,
#    and for Matlab by Alan Genz. Minor bug modifications 7/98, 2/10.
#
#
# Copyright (C) 2013, Alan Genz,  All rights reserved.        
# Translated by ChatGPT
# https://chatgpt.com/share/679ad125-eee8-8010-b44f-5d771f0fa791


# pythran export bvnu(float, float, float)
def bvnu(dh, dk, r):
    if dh == np.inf or dk == np.inf:
        return 0
    elif dh == -np.inf:
        return 1 if dk == -np.inf else phid(-dk)
    elif dk == -np.inf:
        return phid(-dh)
    elif r == 0:
        return phid(-dh) * phid(-dk)
    else:
        tp = 2 * np.pi
        h, k, hk = dh, dk, dh * dk
        bvn = 0

        if abs(r) < 0.3:
            w = np.array([0.1713244923791705, 0.3607615730481384, 0.4679139345726904])
            x = np.array([0.9324695142031522, 0.6612093864662647, 0.2386191860831970])
        elif abs(r) < 0.75:
            w = np.array([0.04717533638651177, 0.1069393259953183, 0.1600783285433464,
                          0.2031674267230659, 0.2334925365383547, 0.2491470458134029])
            x = np.array([0.9815606342467191, 0.9041172563704750, 0.7699026741943050,
                          0.5873179542866171, 0.3678314989981802, 0.1252334085114692])
        else:
            w = np.array([0.01761400713915212, 0.04060142980038694, 0.06267204833410906,
                          0.08327674157670475, 0.1019301198172404, 0.1181945319615184,
                          0.1316886384491766, 0.1420961093183821, 0.1491729864726037,
                          0.1527533871307259])
            x = np.array([0.9931285991850949, 0.9639719272779138, 0.9122344282513259,
                          0.8391169718222188, 0.7463319064601508, 0.6360536807265150,
                          0.5108670019508271, 0.3737060887154196, 0.2277858511416451,
                          0.07652652113349733])

        w = np.concatenate((w, w))
        x = np.concatenate((1 - x, 1 + x))

        if abs(r) < 0.925:
            hs = (h ** 2 + k ** 2) / 2
            asr = np.arcsin(r) / 2
            sn = np.sin(asr * x)
            # bvn = np.exp((sn * hk - hs) / (1 - sn ** 2)) @ w
            bvn = np.sum(np.exp((sn * hk - hs) / (1 - sn ** 2)) * w)
            bvn = bvn * asr / tp + phid(-h) * phid(-k)
        else:
            if r < 0:
                k, hk = -k, -hk

            if abs(r) < 1:
                as_ = 1 - r ** 2
                a = np.sqrt(as_)
                bs = (h - k) ** 2
                asr = -(bs / as_ + hk) / 2
                c = (4 - hk) / 8
                d = (12 - hk) / 80

                if asr > -100:
                    bvn = a * np.exp(asr) * (1 - c * (bs - as_) * (1 - d * bs) / 3 + c * d * as_ ** 2)

                if hk > -100:
                    b = np.sqrt(bs)
                    sp = np.sqrt(tp) * phid(-b / a)
                    bvn -= np.exp(-hk / 2) * sp * b * (1 - c * bs * (1 - d * bs) / 3)

                a /= 2
                xs = (a * x) ** 2
                asr = -(bs / xs + hk) / 2
                ix = asr > -100
                xs = xs[ix]
                sp = (1 + c * xs * (1 + 5 * d * xs))
                rs = np.sqrt(1 - xs)
                ep = np.exp(-(hk / 2) * xs / (1 + rs) ** 2) / rs
                # bvn = (a * ((np.exp(asr[ix]) * (sp - ep)) @ w[ix]) - bvn) / tp
                bvn = (a * np.sum((np.exp(asr[ix]) * (sp - ep)) * w[ix]) - bvn) / tp

            if r > 0:
                bvn += phid(-max(h, k))
            elif h >= k:
                bvn = -bvn
            else:
                L = phid(k) - phid(h) if h < 0 else phid(-h) - phid(-k)
                bvn = L - bvn

        return max(0, min(1, bvn))


# pythran export bvn(float, float, float, float, float)
def bvn(xl, xu, yl, yu, r):
    p = bvnu(xl, yl, r) - bvnu(xu, yl, r) - bvnu(xl, yu, r) + bvnu(xu, yu, r)
    return max(0, min(1, p))


def bvnl(dh, dk, r):
    return bvnu(-dh, -dk, r)

### Trivariate Normal

#TVN
#  A function for computing trivariate normal probabilities.
#    p = tvn( lw, up, cr, epsi )
#  Parameters
#     lw  array of lower integration limits.
#     up  array of upper integration limits.
#     cr   array of correlation coefficients, in order c21, c31, c32.
#     epsi  optional (default = 1e-7) absolute accuracy; highest accuracy
#            for most computations is approximately 1e-14.
#  Example
#   p = tvn( [-3,-4,-inf], [inf,4,3], [.3,-.4,.5] )
#
#
#TVNL
#     A function for computing trivariate normal probabilities.
#     It calculates the probability that x(i) < h(i), for i = 1, 2, 3.
#    h     real array of three upper limits for probability distribution
#    r     real array of three correlation coefficients, r should
#          contain the lower left portion of the correlation matrix R.
#          r should contain the values r21, r31, r32 in that order.
#    epsi  optional (default = 1e-7) absolute accuracy; maximum accuracy
#            for most computations is approximately 1e-14.
#   Example: p = tvnl( [1 2 4], [.3 .4 -.6] )
#

#     This function uses algorithms developed from the ideas
#     described in the papers:
#       R.L. Plackett, Biometrika 41(1954), pp. 351-360.
#       Z. Drezner, Math. Comp. 62(1994), pp. 289-294.
#     with adaptive integration from (0,0,r32) for R.
#     The software is based on work described in the paper
#      "Numerical Computation of Rectangular Bivariate and Trivariate
#        Normal and t Probabilities", by
#          Alan Genz
#          Department of Mathematics
#          Washington State University
#          Pullman, WA 99164-3113
#          Email : alangenz@wsu.edu
#     This file contains functions tvnl (trivariate normal), bvnl
#      (bivariate normal), phid (univariate normal), plus support functions.
#
#   Copyright (C) 2011, Alan Genz,  All rights reserved.


# pythran export tvn(float[:], float[:], float[:], float)
def tvn(lw, up, cr, epsi=1e-7):
    """
    Compute trivariate normal probabilities.
    """
    ep = max(1e-14, epsi)
    lw, up = np.asarray(lw), np.asarray(up)
    lu = np.vstack((lw, up)) if lw.ndim == 1 else np.hstack((lw, up))

    tvn_val = tvnl(up, cr, ep) - tvnl(lw, cr, ep)
    tvn_val -= tvnl([lu[0, 0], lu[1, 1], lu[1, 2]], cr, ep)
    tvn_val -= tvnl([lu[1, 0], lu[0, 1], lu[1, 2]], cr, ep)
    tvn_val -= tvnl([lu[1, 0], lu[1, 1], lu[0, 2]], cr, ep)
    tvn_val += tvnl([lu[0, 0], lu[0, 1], lu[1, 2]], cr, ep)
    tvn_val += tvnl([lu[0, 0], lu[1, 1], lu[0, 2]], cr, ep)
    tvn_val += tvnl([lu[1, 0], lu[0, 1], lu[0, 2]], cr, ep)

    return np.clip(tvn_val, 0, 1)


# pythran export tvnl(float[:], float[:], float)
def tvnl(h, r, epsi=1e-7):
    """
    Compute trivariate normal probability for upper limits h and correlation r.
    """
    epst = max(1e-14, epsi)

    if np.any(np.isneginf(h)):
        return 0.0

    if np.isposinf(h[0]):
        if np.isposinf(h[1]):
            return 1.0 if np.isposinf(h[2]) else phid(h[2])
        return phid(h[1]) if np.isposinf(h[2]) else bvnl(h[1], h[2], r[2])

    if np.isposinf(h[1]):
        return phid(h[0]) if np.isposinf(h[2]) else bvnl(h[0], h[2], r[1])

    if np.isposinf(h[2]):
        return bvnl(h[0], h[1], r[0])

    h1, h2, h3 = h
    r12, r13, r23 = r

    if abs(r12) > abs(r13):
        h2, h3 = h3, h2
        r12, r13 = r13, r12
    if abs(r13) > abs(r23):
        h1, h2 = h2, h1
        r23, r13 = r13, r23

    if abs(h1) + abs(h2) + abs(h3) < epst:
        tvn_val = (1 + 2 * (np.arcsin(r12) + np.arcsin(r13) + np.arcsin(r23)) / np.pi) / 8
    elif abs(r12) + abs(r13) < epst:
        tvn_val = phid(h1) * bvnl(h2, h3, r23)
    elif abs(r13) + abs(r23) < epst:
        tvn_val = phid(h3) * bvnl(h1, h2, r12)
    elif abs(r12) + abs(r23) < epst:
        tvn_val = phid(h2) * bvnl(h1, h3, r13)
    elif 1 - r23 < epst:
        tvn_val = bvnl(h1, min(h2, h3), r12)
    elif r23 + 1 < epst and h2 > -h3:
        tvn_val = bvnl(h1, h2, r12) - bvnl(h1, -h3, r12)
    else:
        a12, a13 = np.arcsin(r12), np.arcsin(r13)
        tvn_val = adonet(lambda x: tvnf(x, h1, h2, h3, r23, a12, a13), 0, 1, epst) / (2 * np.pi)
        tvn_val += bvnl(h2, h3, r23) * phid(h1)

    return max(0, min(tvn_val, 1))


def krnrdt(a, b, f):
    """
    Kronrod integration rule
    """
    wg0 = 0.2729250867779007
    wg = np.array([0.05566856711617449, 0.1255803694649048, 0.1862902109277352,
                   0.2331937645919914, 0.2628045445102478])

    xgk = np.array([0.9963696138895427, 0.9782286581460570, 0.9416771085780681,
                    0.8870625997680953, 0.8160574566562211, 0.7301520055740492,
                    0.6305995201619651, 0.5190961292068118, 0.3979441409523776,
                    0.2695431559523450, 0.1361130007993617])

    wgk0 = 0.1365777947111183
    wgk = np.array([0.00976544104596129, 0.02715655468210443, 0.04582937856442671,
                     0.06309742475037484, 0.07866457193222764, 0.09295309859690074,
                     0.1058720744813894, 0.1167395024610472, 0.1251587991003195,
                     0.1312806842298057, 0.1351935727998845])

    wid = (b - a) / 2
    cen = (b + a) / 2
    fc = f(cen)
    resg = fc * wg0
    resk = fc * wgk0

    for j in range(5):
        t = wid * xgk[2*j]
        fc = f(cen - t) + f(cen + t)
        resk += wgk[2*j] * fc
        t = wid * xgk[2*j+1]
        fc = f(cen - t) + f(cen + t)
        resk += wgk[2*j+1] * fc
        resg += wg[j] * fc

    t = wid * xgk[10]
    fc = f(cen - t) + f(cen + t)
    resk = wid * (resk + wgk[10] * fc)
    err = abs(resk - wid * resg)

    return resk, err


def adonet(f, a, b, tol, nl=100):
    x = np.zeros(nl)
    y = np.zeros(nl)
    err = np.zeros(nl)
    fi = np.zeros(nl)

    x[0], y[0] = a, b
    fi[0], err[0] = krnrdt(a, b, f)
    n = 1

    while 4 * np.sqrt(np.sum(err[:n] ** 2)) > tol and n < nl:
        i = np.argmax(err[:n])
        mid = (x[i] + y[i]) / 2
        x[n], y[n] = mid, y[i]
        y[i] = mid

        fi[i], err[i] = krnrdt(x[i], y[i], f)
        fi[n], err[n] = krnrdt(x[n], y[n], f)
        n += 1

    return np.sum(fi[:n])


def tvnf(x, h1, h2, h3, r23, a12, a13):
    f = 0
    r12, rr2 = sincs(a12 * x)
    r13, rr3 = sincs(a13 * x)

    if abs(a12) > 0:
        f += a12 * pntgnd(h1, h2, h3, r13, r23, r12, rr2)
    if abs(a13) > 0:
        f += a13 * pntgnd(h1, h3, h2, r12, r23, r13, rr3)

    return f


def sincs(x):
    ee = (np.pi / 2 - abs(x)) ** 2
    if ee < 5e-5:
        cs = ee * (1 - ee * (1 - 2 * ee / 15) / 3)
        sx = (1 - ee * (1 - ee / 12) / 2) * np.sign(x)
    else:
        sx = np.sin(x)
        cs = 1 - sx * sx
    return sx, cs


def pntgnd(ba, bb, bc, ra, rb, r, rr):
    dt = rr * (rr - (ra - rb) ** 2 - 2 * ra * rb * (1 - r))
    if dt <= 0:
        return 0
    bt = (bc * rr + ba * (r * rb - ra) + bb * (r * ra - rb)) / np.sqrt(dt)
    ft = (ba - r * bb) ** 2 / rr + bb ** 2
    if bt > -10 and ft < 100:
        f = np.exp(-ft / 2)
        if bt < 10:
            f *= phid(bt)
        return f
    return 0
