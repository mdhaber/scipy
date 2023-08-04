from functools import cached_property
import numpy as np
_null = object()
oo = np.inf


class ContinuousDistribution:
    def __init__(self, **shapes):

        # identify parameterization
        shapes = {key:val for key, val in shapes.items() if val is not _null}
        shape_names, shape_vals = zip(*shapes.items())
        for parameterization in self._parameterizations:
            if parameterization.validate(set(shape_names)):
                break
        else:
            message = (f"The provided shapes `{set(shape_names)}` "
                       "do not match a supported parameterization of the "
                       f"`{self.__class__.__name__}` distribution family.")
            raise ValueError(message)
        self._parameterization = parameterization

        # broadcast shape arguments
        try:
            shape_vals = np.broadcast_arrays(*shape_vals)
        except ValueError as e:
            message = (f"The shapes {set(shape_names)} provided to the "
                       f"`{self.__class__.__name__}` distribution family cannot "
                       "be broadcast to the same shape.")
            raise ValueError(message) from e

        # Compress: remove invalid shapes and ravel
        shapes = dict(zip(shape_names, shape_vals))
        self._valid = parameterization.validate_shapes(shapes)
        shapes = self._compress(**shapes)
        self._shapes = shapes

    def _compress(self, **shapes):
        # return [array[self._valid] for array in arrays]
        return {name: val[self._valid] for name, val in shapes.items()}

    def _decompress(self, *arrays):
        outs = []
        for array in arrays:
            out = np.full(self._valid.shape, np.nan, dtype=array.dtype)
            out[self._valid] = array
            outs.append(out[()])
        return outs

    @cached_property
    def _support(self):
        a, b = self._variable.domain.endpoints
        a = getattr(self, "_"+a, a)
        b = getattr(self, "_"+b, b)
        return a, b

    @property
    def support(self):
        return self._decompress(*self._support)

    @classmethod
    def _draw(cls, sizes=None, rng=None, i_parameterization=0):
        parameterization = cls._parameterizations[i_parameterization]
        shapes = parameterization.draw(sizes, rng)
        return cls(**shapes)


class _Domain:
    symbols = {np.inf: "∞", -np.inf: "-∞", np.pi: "π", -np.pi: "-π"}

    def contains(self, x):
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()


class _SimpleDomain(_Domain):

    def define_parameters(self, *parameters):
        new_symbols = {param.name: param.symbol for param in parameters}
        self.symbols.update(new_symbols)

    def contains(self, item, shapes={}):
        a, b = self.endpoints
        left_inclusive, right_inclusive = self.inclusive

        a = shapes.get(a, a)
        b = shapes.get(b, b)

        in_left = item >= a if left_inclusive else item > a
        in_right = item <= b if right_inclusive else item < b
        return in_left & in_right


class _RealDomain(_SimpleDomain):

    def __init__(self, endpoints=(-oo, oo), inclusive=(False, False)):
        self.endpoints = endpoints
        self.inclusive = inclusive

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

    def __str__(self):
        return f"Accepts `{self.name}` for ${self.symbol} ∈ {str(self.domain)}$."

    def draw(self, size=None, rng=None):
        rng = rng or np.random.default_rng()
        return rng.uniform(*self.typical, size=size)


class _RealParameter(_Parameter):
    def __init__(self, name, *, typical=None, symbol=None, domain=_RealDomain()):
        typical = typical or domain
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
    def __init__(self, name, *, typical=None, symbol=None, domain=_IntegerDomain()):
        typical = typical or domain
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
        return arr, valid_dtype


class _Parameterization:
    def __init__(self, *parameters):
        self.parameters = {param.name: param for param in parameters}

    def validate(self, shapes):
        return shapes == set(self.parameters.keys())

    def validate_shapes(self, shapes):
        all_valid = True
        for name, arr in shapes.items():
            parameter = self.parameters[name]
            arr, valid = parameter.check_dtype(arr)
            valid = valid & parameter.domain.contains(arr, shapes)
            all_valid = all_valid & valid
            shapes[name] = arr

        return all_valid

    def __str__(self):
        messages = [str(param) for name, param in self.parameters.items()]
        return " ".join(messages)

    def draw(self, sizes=None, rng=None):
        shapes = {}
        sizes = sizes if np.iterable(sizes) else [sizes]*len(self.parameters)
        for size, param in zip(sizes, self.parameters.values()):
            shapes[param.name] = param.draw(size, rng)
        return shapes


class LogUniform(ContinuousDistribution):

    _a_domain = _RealDomain(endpoints=(0, oo))
    _b_domain = _RealDomain(endpoints=('a', oo))
    _log_a_domain = _RealDomain(endpoints=(-oo, oo))
    _log_b_domain = _RealDomain(endpoints=('log_a', oo))
    _x_support = _RealDomain(endpoints=('a', 'b'))

    _a_param = _RealParameter('a', domain=_a_domain, typical=(1e-3, 1))
    _b_param = _RealParameter('b', domain=_b_domain, typical=(1, 1e3))
    _log_a_param = _RealParameter('log_a', symbol=r'\log(a)',
                                  domain=_log_a_domain, typical=(-3, 0))
    _log_b_param = _RealParameter('log_b', symbol=r'\log(b)',
                                  domain=_log_b_domain, typical=(0, 3))
    _x_param = _RealParameter('x', domain=_x_support)

    _b_domain.define_parameters(_a_param)
    _log_b_domain.define_parameters(_log_a_param)

    _parameterizations = [_Parameterization(_log_a_param, _log_b_param),
                          _Parameterization(_a_param, _b_param)]
    _variable = _x_param

    def __init__(self, *, a=_null, b=_null, log_a=_null, log_b=_null):
        super().__init__(a=a, b=b, log_a=log_a, log_b=log_b)

    @cached_property
    def _a(self):
        return (self._shapes['a'] if 'a' in self._shapes
                else np.exp(self._shapes['log_a']))

    @cached_property
    def _b(self):
        return (self._shapes['b'] if 'b' in self._shapes
                else np.exp(self._shapes['log_b']))

    @cached_property
    def _log_a(self):
        return (self._shapes['log_a'] if 'log_a' in self._shapes
                else np.log(self._shapes['a']))

    @cached_property
    def _log_b(self):
        return (self._shapes['log_b'] if 'log_b' in self._shapes
                else np.log(self._shapes['b']))
