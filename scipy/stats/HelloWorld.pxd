cdef extern from "HelloWorld.cpp":
    pass

cdef extern from "HelloWorld.h":
    int hello_world(double* x, int n)