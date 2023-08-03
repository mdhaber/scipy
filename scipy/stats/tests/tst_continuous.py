import numpy as np
import pytest
import hypothesis

from scipy.stats._distribution_infrastructure import _ShapeInfo, oo, LogUniform

# _a_info = _ShapeInfo('a', domain=(0, oo), inclusive=(False, False),
#                      typical=(1e-3, 1e-2))
# _b_info = _ShapeInfo('b', domain=(0, oo), inclusive=(False, False),
#                      typical=(1e2, 1e3))
#
# parameterization = _Parameterization(_a_info, _b_info)
#
shapes = {'a': 1, 'log_a': 2}

LogUniform(**shapes)