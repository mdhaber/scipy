cdef extern from "helloworld/HelloWorld.cpp":
    pass

cdef extern from "helloworld/HelloWorld.h":
    int hello_world(double* x, int n)