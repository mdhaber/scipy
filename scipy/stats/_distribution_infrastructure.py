import numpy as np
_null = object()
oo = np.inf


class ContinuousDistribution:
    def __init__(self, **shapes):

        # determine parameterization
        shape_names = set((key for key, val in shapes.items()
                           if val is not _null))
        for parameterization in self._parameterizations:
            if parameterization.validate(shape_names):
                break
        else:
            message = (f"The provided shapes {shape_names} "
                       "do not match a supported parameterization.")
            raise ValueError(message)

        original_shapes, standard_shapes, valid = parameterization.validate_shapes(shapes)


class _Domain:
    symbols = {np.inf: "∞", -np.inf: "-∞", np.pi: "π", -np.pi: "-π"}

    def __contains__(self, x):
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()


class _SimpleDomain(_Domain):
    def __init__(self, endpoints=(-oo, oo), inclusive=(False, False)):
        self.endpoints = endpoints
        self.includes = inclusive

    def __contains__(self, item):
        a, b = self.endpoints
        left_inclusive, right_inclusive = self.inclusive

        in_left = item >= a if left_inclusive else item > a
        in_right = item <= b if right_inclusive else item < b
        return in_left & in_right


class _RealDomain(_SimpleDomain):

    def __str__(self):
        a, b = self.endpoints
        left_inclusive, right_inclusive = self.inclusive

        left = "[" if left_inclusive else "("
        a = self.symbols.get(a, f"{a}")
        right = "]" if right_inclusive else ")"
        b = self.symbols.get(b, f"{b}")

        return f"{left}{a}, {b}{right}"


class _IntegerDomain(_SimpleDomain):
    pass


class _Parameter:
    def __init__(self, name, *, symbol, domain, typical):
        self.name = name
        self.symbol = symbol or name
        self.domain = domain
        self.typical = typical

class _RealParameter(_Parameter):
    def __init__(self, name, *, typical, symbol=None, domain=_RealDomain()):
        symbol = symbol or name
        super().__init__(name, symbol=symbol, domain=domain, typical=typical)

    def check_dtype(self, arr):
        arr = np.asarray(arr)
        dtype = arr.dtype
        valid_dtype = np.ones_like(arr, dtype=bool)
        if np.issubdtype(dtype, np.floating):
            pass
        elif np.issubdtype(dtype, np.integer):
            dtype = np.float64
            arr = np.asarray(arr, dtype=dtype)
        elif np.issubdtype(dtype, np.complexfloating):
            real_arr = np.real(arr)
            valid_dtype = (real_arr == arr)
            arr = real_arr
        else:
            message = f"Parameter {self.name} must be of real dtype."
            raise ValueError(message)
        return arr, valid_dtype

class _IntegerParameter(_Parameter):
    def __init__(self, name, *, typical, symbol=None, domain=_IntegerDomain()):
        symbol = symbol or name
        super().__init__(name, symbol=symbol, domain=domain, typical=typical)

    def check_dtype(self, arr):
        arr = np.asarray(arr)
        dtype = arr.dtype
        valid_dtype = np.ones_like(arr, dtype=bool)
        if np.issubdtype(dtype, np.integer):
            pass
        elif np.issubdtype(dtype, np.inexact):
            integral_arr = np.round(arr)
            valid_dtype = (integral_arr == arr)
            arr = integral_arr
        else:
            message = f"Parameter {self.name} must be of integer dtype."
            raise ValueError(message)
        return arr


class _Parameterization:
    def __init__(self, *parameters):
        self.parameters = {param.name: param for param in parameters}
        # self.integer_parameters = {param.name: param for param in parameters
        #                            if isinstance(param, _IntegerShape)}
        # self.real_parameters = {param.name: param for param in parameters
        #                         if isinstance(param, _RealShape)}

    def validate(self, shapes):
        return shapes == set(self.parameters.keys())

    def validate_shapes(self, **original_shapes):
        # check dtypes
        # check simple bounds
        # convert to canonical parameterization
        # broadcast and reshape

        standard_shapes = {}
        valid_shapes = {}
        for name, arr in original_shapes.items():
            parameter = self.parameters[name]
            arr, valid = parameter.check_dtype(arr)
            valid &= arr in parameter.domain
            standard_shapes[name] = arr
            valid_shapes[name] = valid

        standard_shapes = self.canonical_shapes(standard_shapes)

        return original_shapes, standard_shapes, valid_shapes

    def canonical_shapes(self, **shapes):
        return shapes


class _LogUniformStandard(_Parameterization):
    def canonical_shapes(self, a, b):
        return dict(a=np.log(a), b=np.log(b))


class LogUniform(ContinuousDistribution):
    _a_info = _Parameter('a', domain=(0, oo), inclusive=(False, False), typical=(1e-3, 1e-2))
    _b_info = _Parameter('b', domain=(0, oo), inclusive=(False, False), typical=(1e2, 1e3))
    _log_a_info = _Parameter('log_a', symbol=r'\log(a)', inclusive=(False, False), typical=(-3, -2))
    _log_b_info = _Parameter('log_b', symbol=r'\log(b)', inclusive=(False, False), typical=(2, -3))
    _log_parameterization = _Parameterization(_log_a_info, _log_b_info)
    _parameterizations = [_LogUniformStandard, _log_parameterization]

    def __init__(self, *, a=_null, b=_null, log_a=_null, log_b=_null):
        super().__init__(a=a, b=b, log_a=log_a, log_b=log_b)
