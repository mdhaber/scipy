# distutils: language = c++

from .BiasedUrn cimport CFishersNCHypergeometric, StochasticLib3

cdef class _PyFishersNCHypergeometric:
    cdef CFishersNCHypergeometric c_fnch

    def __cinit__(self, int n, int m, int N, double odds, double accuracy):
        self.c_fnch = CFishersNCHypergeometric(n, m, N, odds, accuracy)

    def mode(self):
        return self.c_fnch.mode()
        
    def mean(self):
        return self.c_fnch.mean()
        
    def variance(self):
        return self.c_fnch.variance()
        
    def probability(self, int x):
        return self.c_fnch.probability(x)

    def moments(self):
        cdef double mean, var
        self.c_fnch.moments(&mean, &var)
        return mean, var

cdef class _PyStochasticLib3:
    cdef StochasticLib3 c_sl3

    def __cinit__(self, int seed):
        self.c_sl3 = StochasticLib3(seed)
                
    def SetAccuracy(self, double accur):
        return self.c_sl3.SetAccuracy(accur)

    def FishersNCHyp(self, int n, int m, int N, double odds):
        return self.c_sl3.FishersNCHyp(n, m, N, odds)
