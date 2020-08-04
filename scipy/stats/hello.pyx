# distutils: language = c++

from .HelloWorld cimport hello_world
import numpy as np

def py_hello_world(double[::1] x):
    hello_world(&x[0], x.shape[0])