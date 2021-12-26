"""
Unit test for Mixed Integer Linear Programming
"""

import re
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from .test_linprog import magic_square
from scipy.optimize import milp, Bounds, LinearConstraint


def test_milp_iv():

    message = "`c` must be a one-dimensional array"
    with pytest.raises(ValueError, match=message):
        milp(np.zeros((3, 4)))

    message = "`bounds` must be an instance of..."
    with pytest.raises(ValueError, match=message):
        milp(1, bounds=10)

    message = "`constraints` must be an instance of..."
    with pytest.raises(ValueError, match=message):
        milp(1, constraints=10)

    message = "`b_l` and `b_u` must be one-dimensional."
    with pytest.raises(ValueError, match=message):
        milp(np.zeros(3), constraints=([[1, 2, 3]], [[2]], [2]))
    with pytest.raises(ValueError, match=message):
        milp(np.zeros(3), constraints=([[1, 2, 3]], [2], [[2]]))

    message = "The shape of `A` must be (len(b_l), len(c))."
    with pytest.raises(ValueError, match=re.escape(message)):
        milp(np.zeros(3), constraints=([[1, 2, 3]], [2, 3], [2, 3]))
    with pytest.raises(ValueError, match=re.escape(message)):
        milp(np.zeros(3), constraints=([[1, 2]], [2], [2]))

    message = "operands could not be broadcast together"
    with pytest.raises(ValueError, match=message):
        milp([1, 2, 3], integrality = [1, 2])
    with pytest.raises(ValueError, match=message):
        milp([1, 2, 3], bounds=([1, 2], [3, 4, 5]))
    with pytest.raises(ValueError, match=message):
        milp([1, 2, 3], bounds=([1, 2, 3], [4, 5]))


def test_milp_optional_args():
    # check that arguments other than `c` are indeed optional
    res = milp(1)
    assert res.fun == 0
    assert_array_equal(res.x, [0])


def test_milp_1():
    # solve magic square problem
    n = 3
    A, b, c, numbers, M = magic_square(n)
    res = milp(c=c*0, constraints=(A, b, b), bounds=(0, 1), integrality=1)

    # check that solution is a magic square
    x = np.round(res.x)
    s = (numbers.flatten() * x).reshape(n**2, n, n)
    square = np.sum(s, axis=0)
    np.testing.assert_allclose(square.sum(axis=0), M)
    np.testing.assert_allclose(square.sum(axis=1), M)
    np.testing.assert_allclose(np.diag(square).sum(), M)
    np.testing.assert_allclose(np.diag(square[:, ::-1]).sum(), M)


def test_milp_2():
    # solve MIP with inequality constraints and all integer constraints
    # source: slide 5,
    # https://www.cs.upc.edu/~erodri/webpage/cps/theory/lp/milp/slides.pdf
    c = -np.ones(2)
    A = [[-2, 2], [-8, 10]]
    b_l = [1, -np.inf]
    b_u = [np.inf, 13]

    # solve original problem
    res = milp(c=c, constraints=(A, b_l, b_u), integrality=True)
    np.testing.assert_allclose(res.x, [1, 2])
    np.testing.assert_allclose(res.fun, -3)

    # solve relaxed problem
    res = milp(c=c, constraints=(A, b_l, b_u))
    np.testing.assert_allclose(res.x, [4, 4.5])
    np.testing.assert_allclose(res.fun, -8.5)


def test_milp_3():
    # solve MIP with inequality constraints and all integer constraints
    # source: https://en.wikipedia.org/wiki/Integer_programming#Example
    c = [0, -1]
    A = [[-1, 1], [3, 2], [2, 3]]
    b_u = [1, 12, 12]
    b_l = np.full_like(b_u, -np.inf)
    constraints = LinearConstraint(A, b_l, b_u)

    integrality = np.ones_like(c)

    # solve original problem
    res = milp(c=c, constraints=constraints, integrality=integrality)
    assert res.message == 'Optimal'
    assert_allclose(res.fun, -2)
    assert_allclose(res.x, [1, 2])

    # solve relaxed problem
    res = milp(c=c, constraints=constraints)
    assert res.message == 'Optimal'
    assert_allclose(res.fun, -2.8)
    assert_allclose(res.x, [1.8, 2.8])


def test_milp_4():
    # solve MIP with inequality constraints and only one integer constraint
    # source: https://www.mathworks.com/help/optim/ug/intlinprog.html
    c = [8, 1]
    integrality = [0, 1]
    A = [[1, 2], [-4, -1], [2, 1]]
    b_l = [-14, -np.inf, -np.inf]
    b_u = [np.inf, -33, 20]
    constraints = LinearConstraint(A, b_l, b_u)
    bounds = Bounds(-np.inf, np.inf)

    res = milp(c, integrality=integrality, bounds=bounds,
               constraints=constraints)
    assert res.message == 'Optimal'
    assert_allclose(res.fun, 59)
    assert_allclose(res.x, [6.5, 7])


def test_milp_5():
    # solve MIP with inequality and equality constraints
    # source: https://www.mathworks.com/help/optim/ug/intlinprog.html
    c = [-3, -2, -1]
    integrality = [0, 0, 1]
    lb = [0, 0, 0]
    ub = [np.inf, np.inf, 1]
    bounds = Bounds(lb, ub)
    A = [[1, 1, 1], [4, 2, 1]]
    b_l = [-np.inf, 12]
    b_u = [7, 12]
    constraints = LinearConstraint(A, b_l, b_u)

    res = milp(c, integrality=integrality, bounds=bounds,
               constraints=constraints)
    # there are multiple solutions
    assert_allclose(res.fun, -12)


@pytest.mark.slow
def test_milp_6():
    # solve a larger MIP with only equality constraints
    # source: https://www.mathworks.com/help/optim/ug/intlinprog.html
    # TODO: figure out why this intermittently segfaults!
    integrality = 1
    A_eq = np.array([[22, 13, 26, 33, 21,  3, 14, 26],
                     [39, 16, 22, 28, 26, 30, 23, 24],
                     [18, 14, 29, 27, 30, 38, 26, 26],
                     [41, 26, 28, 36, 18, 38, 16, 26]])
    b_eq = np.array([7872, 10466, 11322, 12058])
    c = np.array([2, 10, 13, 17, 7, 5, 7, 3])

    res = milp(c=c, constraints=(A_eq, b_eq, b_eq), integrality=integrality)

    np.testing.assert_allclose(res.fun, 1854)
