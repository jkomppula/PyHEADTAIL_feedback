import numpy as np
from scipy.constants import pi

class Averager(object):
    """The simplest possible signal mixer, which calculates a phase weighted average of
    signals from different registers. Readings x/y axis are converted to xp/yp axis by adding pi/2 phase sift to
    betatron phase angle and multiplying values with phase_conv_coeff which describes an amplitude scaling between
    x/y and xp/yp."""

    def __init__(self, phase_conv_coeff):
        self.phase_conv_coeff = phase_conv_coeff

    def mix(self,registers,reader_phase_angle):

        signal = None

        for index, register in enumerate(registers):
            if signal is None:
                signal = register.read_signal(reader_phase_angle+pi/2.)
            else:
                signal += register.read_signal(reader_phase_angle+pi/2.)
            signal = self.phase_conv_coeff*signal/float(len(registers))
        return signal

class RightAnglePickups(object):
    def __init__(self, register_1, register_2):
        self.register_1 = register_1
        self.register_2 = register_2

    def mix(self):
        avg_length = min(self.register_1.avg_length,self.register_2.avg_length)

        signal = None
        for idx in range(avg_length):
            val_1 = self.register_1.read_value(idx)
            val_2 = self.register_2.read_value(idx)

            if idx == 0:
                signal = np.zeros(len(val_1))

            if (val_1 is not None) and (val_2 is not None):
                signal += np.sqrt(val_1*val_1 + val_2*val_2)
            else:
                break

        return signal


class DoublePickup(object):
    def __init__(self, register_1,register_2):
        self.register_1 = register_1
        self.register_2 = register_2

    def mix(self):
        return 1