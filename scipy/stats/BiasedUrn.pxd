# Declare the class with cdef
cdef extern from "biasedurn/stocc.h":
    cdef cppclass CFishersNCHypergeometric:
        CFishersNCHypergeometric() except +
        CFishersNCHypergeometric(int, int, int, double, double) except +
        int mode()
        double mean()
        double variance()
        double probability(int x)
        double moments(double * mean, double * var)

    cdef cppclass StochasticLib3:
        StochasticLib3() except +
        StochasticLib3(int seed) except +
        void SetAccuracy(double accur);
        int FishersNCHyp (int n, int m, int N, double odds)