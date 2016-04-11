import numpy as np
import itertools
import math
from scipy.constants import pi

class Averager(object):
    """The simplest possible signal mixer of pick ups, which calculates a phase weighted average of
    the pick up signals"""

    def __init__(self, x_xp_conv_coeff):
        self.x_xp_conv_coeff = x_xp_conv_coeff

    def mix(self,registers,reader_phase_angle):

        signal = None

        for index, register in enumerate(registers):
            if signal is None:
                #signal = register.read_signal(reader_phase_angle+0.)
                signal = register.read_signal(reader_phase_angle+pi/2.)
            else:
                #signal += register.read_signal(reader_phase_angle+0.)
                signal += register.read_signal(reader_phase_angle+pi/2.)
            signal = self.x_xp_conv_coeff*signal/float(len(registers))
        return signal
