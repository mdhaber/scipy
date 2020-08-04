# distutils: language = c++

from .HelloWorld cimport hello_world

def py_hello_world():
    hello_world()