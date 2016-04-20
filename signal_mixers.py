import numpy as np
from scipy.constants import pi

class Averager(object):
    """The simplest possible signal mixer, which calculates an average of
    signals from different registers. If x_to_xp is True, readings are converted from x/y axis to to xp/yp axis by
    adding pi/2 to the position phase angle and multiplying values with phase_conv_coeff.
    phase_conv_coeff describes an amplitude scaling between x/y and xp/yp."""

    def __init__(self, phase_conv_coeff, x_to_xp = True):
        self.phase_conv_coeff = phase_conv_coeff
        self.x_to_xp = x_to_xp

    def mix(self,registers,reader_position):

        total_signal = None
        n_signals = 0

        if self.x_to_xp == True:
            reader_position += pi/2

        for register in registers:
            for signal in register(reader_position):
                if total_signal is None:
                    total_signal = np.zeros(len(signal))

                n_signals += 1
                total_signal += signal

        total_signal /= float(n_signals)

        if self.x_to_xp == True:
            total_signal *= self.phase_conv_coeff

        return total_signal


# class RightAnglePickups(object):
#     # Assumes that betatron phase difference between pickups is pi/2 (+ n*2 pi). Thus, the betatron amplitude is
#     # a quadratic sum of signals from the pickups independently of the betatron phase angle
#
#     def __init__(self, register_1,register_2):
#         self.register_1 = register_1
#         self.register_2 = register_2
#
#     def mix(self):
#         total_signal = None
#
#         n_signals = min(len(self.register_1),len(self.register_2))
#
#         for signal_1, signal_2 in zip(self.register_1,self.register_2):
#
#             if total_signal is None:
#                 total_signal = np.zeros(len(signal_1))
#
#             total_signal += np.sqrt(signal_1*signal_1+signal_2*signal_2)/float(n_signals)
#
#         return total_signal
