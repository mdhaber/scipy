import numpy as np
import pytest
import hypothesis

from scipy.stats._distribution_infrastructure import oo, LogUniform

# _a_info = _ShapeInfo('a', domain=(0, oo), inclusive=(False, False),
#                      typical=(1e-3, 1e-2))
# _b_info = _ShapeInfo('b', domain=(0, oo), inclusive=(False, False),
#                      typical=(1e2, 1e3))
#
# parameterization = _Parameterization(_a_info, _b_info)
#
a = [-1, 1]
b = [0.5, 1.5]
x = [[0.5], [1.25], [2]]
# dist = LogUniform(a=a, b=b)
# dist = LogUniform(log_a=np.log(a), log_b=np.log(b))
# x = np.zeros((3, 0))
dist = LogUniform(a=a, b=b)

# print(dist.shapes['a'])
# print(dist.shapes['b'])
# print(dist.shapes_in_domain)
print(dist._parameterization)
a, b = dist.support
print(a)
print(b)
#
# dist.pdf(1)


# from scipy import stats
# stats.loguniform.pdf(-1, [-1, 1], [2, 3])

# import numpy as np
# from functools import cached_property
#
# class A:
#     @cached_property
#     def f(self):
#         return np.zeros(3)
#
