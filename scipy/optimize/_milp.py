import numpy as np
from scipy.sparse import csc_array
from ._highs._highs_wrapper import _highs_wrapper
from ._constraints import LinearConstraint, Bounds
from ._optimize import OptimizeResult


def _milp_iv(c, integrality, bounds, constraints, options):

    c = np.atleast_1d(c).astype(np.double)
    if c.ndim != 1:
        message = "`c` must be a one-dimensional array."
        raise ValueError(message)

    if integrality is None:
        integrality = 0
    integrality = np.broadcast_to(integrality, c.shape).astype(np.uint8)

    if bounds is None:
        bounds = Bounds(0, np.inf)
    elif not isinstance(bounds, Bounds):
        message = "`bounds` must be an instance of `scipy.optimize.Bounds`."
        try:
            bounds = Bounds(*bounds)
        except TypeError:
            raise ValueError(message)
    # broadcast in case lb/ub are scalars
    lb = np.broadcast_to(bounds.lb, c.shape).astype(np.double)
    ub = np.broadcast_to(bounds.ub, c.shape).astype(np.double)

    if constraints is None:
        constraints = LinearConstraint(np.empty((0, c.size)),
                                       np.empty((0,)), np.empty((0,)))
    elif not isinstance(constraints, LinearConstraint):
        message = ("`constraints` must be an instance of "
                   "`scipy.optimize.LinearConstraint`.")
        try:
            constraints = LinearConstraint(*constraints)
        except TypeError:
            raise ValueError(message)
    A = csc_array(constraints.A)
    b_l = np.atleast_1d(constraints.lb).astype(np.double)
    b_u = np.atleast_1d(constraints.ub).astype(np.double)
    if b_l.ndim > 1 or b_l.shape != b_u.shape:
        message = "`b_l` and `b_u` must be one-dimensional of equal length."
        raise ValueError(message)
    if A.shape != (b_l.size, c.size):
        message = "The shape of `A` must be (len(b_l), len(c))."
        raise ValueError(message)
    indptr, indices, data = A.indptr, A.indices, A.data.astype(np.double)

    # relying on options input validation in _highs_wrapper
    options_iv = {}
    options_iv.update(options)

    return c, integrality, lb, ub, indptr, indices, data, b_l, b_u, options_iv


def milp(c, *, integrality=None, bounds=None, constraints=None, options={}):
    r"""
    Mixed-integer linear programming

    Solves problems of the following form:

    .. math::

        \min_x \ & c^T x \\
        \mbox{such that} \ & b_l \leq A x \leq b_u,\\
        & l \leq x \leq u, \\
        & x_i \in \mathbb{Z}, i \in X_i

    where :math:`x` is a vector of decision variables;
    :math:`c`, :math:`b_l`, :math:`b_u`, :math:`l`, and :math:`u` are vectors;
    :math:`A` is a matrix, and :math:`X_i` is the set of indices of
    decision variables that must be integral.

    Alternatively, that's:

    minimize::

        c @ x

    such that::

        b_l <= A @ x <= b_u
        l <= x <= u
        Specified elements of x must be integers

    By default, ``l = 0`` and ``u = np.inf`` unless specified with
    ``bounds``.

    Parameters
    ----------
    c : 1D array_like
        The coefficients of the linear objective function to be minimized.
        `c` is converted to a double precision array before the problem is
        solved.
    integrality : 1D array_like, optional
        Indicates the type of integrality constraint on each decision variable.

        ``0`` : Continuous variable; no integrality constraint.

        ``1`` : Integer variable; decision variable must be an integer
        within `bounds`.

        ``2`` : Semi-continuous variable; decision variable must be within
        `bounds` or take value ``0``.

        ``3`` : Semi-integer variable; decision variable must be an integer
        within `bounds` or take value ``0``.

        By default, all variables are continuous. `integrality` is converted
        to an array of integers before the problem is solved.
    bounds : scipy.optimize.Bounds, optional
        Bounds on the decision variables. Lower and upper bounds are converted
        to double precision arrays before the problem is solved. The
        ``keep_feasible`` parameter of the `Bounds` object is ignored. If
        not specified, all decision variables are constrained to be
        non-negative.
    constraints : scipy.optimize.LinearConstraint, optional
        Linear constraints of the optimization problem. Before the
        problem is solved, all values are converted to double precision, and
        the matrix of constraint coefficients is converted to an instance
        of `scipy.sparse.csc_matrix`. The ``keep_feasible``
        parameter of the `LinearConstraint` object is ignored.

    Options
    -------
    presolve : bool (default: ``True``)
        Presolve attempts to identify trivial infeasibilities,
        identify trivial unboundedness, and simplify the problem before
        sending it to the main solver. It is generally recommended
        to keep the default setting ``True``; set to ``False`` if
        presolve is to be disabled.
    time_limit : float
        The maximum time in seconds allotted to solve the problem;
        default is the largest possible value for a ``double`` on the
        platform.
    dual_feasibility_tolerance : double (default: 1e-07)
        Dual feasibility tolerance for
        :ref:`'highs-ds' <optimize.linprog-highs-ds>`.
        The minimum of this and ``primal_feasibility_tolerance``
        is used for the feasibility tolerance of
        :ref:`'highs-ipm' <optimize.linprog-highs-ipm>`.
    primal_feasibility_tolerance : double (default: 1e-07)
        Primal feasibility tolerance for
        :ref:`'highs-ds' <optimize.linprog-highs-ds>`.
        The minimum of this and ``dual_feasibility_tolerance``
        is used for the feasibility tolerance of
        :ref:`'highs-ipm' <optimize.linprog-highs-ipm>`.
    ipm_optimality_tolerance : double (default: ``1e-08``)
        Optimality tolerance for
        :ref:`'highs-ipm' <optimize.linprog-highs-ipm>`.
        Minimum allowable value is 1e-12.
    simplex_dual_edge_weight_strategy : str (default: None)
        Strategy for simplex dual edge weights. The default, ``None``,
        automatically selects one of the following.

        ``'dantzig'`` uses Dantzig's original strategy of choosing the most
        negative reduced cost.

        ``'devex'`` uses the strategy described in [15]_.

        ``steepest`` uses the exact steepest edge strategy as described in
        [16]_.

        ``'steepest-devex'`` begins with the exact steepest edge strategy
        until the computation is too costly or inexact and then switches to
        the devex method.

        Curently, ``None`` always selects ``'steepest-devex'``, but this
        may change as new options become available.

    Returns
    -------
    res : OptimizeResult
        An instance of :class:`scipy.optimize.OptimizeResult`. The object
        is guaranteed to have the following attributes.

        status : int
            An integer representing the exit status of the algorithm.
        message : str
            A string descriptor of the exit status of the algorithm.

        Depending on the exit status, the following attributes may
        also be available.

        x : ndarray
            The values of the decision variables that minimizes the
            objective function while satisfying the constraints.
        fun : float
            The optimal value of the objective function ``c @ x``.
        simplex_nit : int
            The total number of iterations performed by the simplex algorithm
            in all phases.
        ipm_nit : int
            The number of iterations performed by the interior-point algorithm.
            This does not include crossover iterations.
        crossover_nit : int
            The number of primal/dual pushes performed after completion of the
            interior-point algorithm.

    Notes
    -----
    `milp` is a wrapper of the HiGHS linear optimization software [1]_.

    References
    ----------
    .. [1] Huangfu, Q., Galabova, I., Feldmeier, M., and Hall, J. A. J.
           "HiGHS - high performance software for linear optimization."
           Accessed 12/25/2021 at https://www.maths.ed.ac.uk/hall/HiGHS/#guide
    .. [2] Huangfu, Q. and Hall, J. A. J. "Parallelizing the dual revised
           simplex method." Mathematical Programming Computation, 10 (1),
           119-142, 2018. DOI: 10.1007/s12532-017-0130-5

    Examples
    --------
    Consider the problem at
    https://en.wikipedia.org/wiki/Integer_programming#Example, which is
    expressed as a maximization problem of two variables. Since `milp` requires
    that the problem be expressed as a minimization problem, the objective
    function coefficients on the decision variables are:

    >>> c = -np.array([0, 1])

    Note the negative sign: we maximize the original objective function
    by minimizing the negative of the objective function.

    We collect the coefficients of the constraints into arrays like:

    >>> A = np.array([[-1, 1], [3, 2], [2, 3]])
    >>> b_u = np.array([1, 12, 12])
    >>> b_l = np.full_like(b_u, -np.inf)

    Because there is no lower limit on these constraints, we have defined a
    variable ``b_l`` full of values representing negative infinity. This may
    be unfamiliar to users of `scipy.optimize.linprog`, which only accepts
    "less than" (or "upper bound") inequality constraints of the form
    ``A_ub @ x <= b_u``. By accepting both ``b_l`` and ``b_u`` of constraints
    ``b_l <= A_ub @ x <= b_u``, `milp` makes it easy to specify "greater than"
    inequality constraints, "less than" inequality constraints, and equality
    constraints concisely.

    These arrays are collected into a single `LinearConstraint` object like:

    >>> from scipy.optimize import LinearConstraint
    >>> constraints = LinearConstraint(A, b_l, b_u)

    The non-negativity bounds on the decision variables are enforced by
    default, so we do not need to provide an argument for `bounds`.

    Finally, the problem states that both decision variables must be integers:

    >>> integrality = np.ones_like(c)

    We solve the problem like:
    >>> from scipy.optimize import milp
    >>> res = milp(c=c, constraints=constraints, integrality=integrality)
    >>> res.x
    [1.0, 2.0]

    Note that had we solved the relaxed problem (without integrality
    constraints):
    >>> res = milp(c=c, constraints=constraints)  # OR:
    >>> # from scipy.optimize import linprog; res = linprog(c, A, b_u)
    >>> res.x
    [1.8, 2.8]

    we would not have obtained the correct solution by rounding to the nearest
    integers.

    """

    args_iv = _milp_iv(c, integrality, bounds, constraints, options)
    c, integrality, lb, ub, indptr, indices, data, b_l, b_u, options = args_iv

    res = _highs_wrapper(c, indptr, indices, data, b_l, b_u,
                         lb, ub, integrality, options)

    return OptimizeResult(res)
