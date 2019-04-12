#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 22:18:41 2019

@author: matthaberland
"""

import numpy as np
from scipy.optimize import linprog
import scipy.io

c = np.array([1, 0, 0, 0, 0, 0, 0])
A_ub = -np.identity(7)
b_ub = np.array([[-2], [-2], [-2], [-2], [-2], [-2], [-2]])
A_eq = np.array([
    [1, 1, 1, 1, 1, 1, 0],
    [0.3, 1.3, 0.9, 0, 0, 0, -1],
    [0.3, 0, 0, 0, 0, 0, -2/3],
    [0, 0.65, 0, 0, 0, 0, -1/15],
    [0, 0, 0.3, 0, 0, 0, -1/15]
])
b_eq = np.array([[100], [0], [0], [0], [0]])
bounds = None
res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method='interior-point')

alldata = scipy.io.loadmat('data_8174_2.mat')