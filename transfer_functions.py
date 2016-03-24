import numpy as np
from scipy.constants import c, e
import scipy.integrate as integrate
import scipy.special as special
import itertools


def phase_linearized_lowpass(f_cutoff):
    def K0(dz):
        return special.k0(abs(f_cutoff*dz/c))
    return K0


def lowpass(f_cutoff):
    def exp(dz):
        if dz < 0:
            return 0
        else:
            return np.exp(-1.*dz*f_cutoff/c)
    return exp