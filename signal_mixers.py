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

        #if self.x_to_xp == True:
        #    reader_position += pi/2
        # TODO: if only two registers, no loop

        if len(registers)>1:

            prev_register = registers[-1]
            for register in registers:
                for signal_1, signal_2 in zip(prev_register,register):
                    if total_signal is None:
                        total_signal = np.array([np.zeros(len(signal_1[0])),np.zeros(len(signal_1[0]))])
                    temp_signal = prev_register.combine(signal_1,signal_2,reader_position,x_to_xp = self.x_to_xp)
                    if temp_signal[1] is not None:
                        total_signal = total_signal + temp_signal
                    else:
                        total_signal[0] = total_signal[0] + temp_signal[0]
                    n_signals += 1
                prev_register = register

        # if len(registers) == 2:
        #
        #     for signal_1, signal_2 in zip(registers[0], registers[1]):
        #         if total_signal is None:
        #             total_signal = np.array([np.zeros(len(signal_1[0])), np.zeros(len(signal_1[0]))])
        #         temp_signal = registers[0].combine(signal_1, signal_2, reader_position, x_to_xp=self.x_to_xp)
        #         if temp_signal[1] is not None:
        #             total_signal = total_signal + temp_signal
        #         else:
        #             total_signal[0] = total_signal[0] + temp_signal[0]
        #         n_signals += 1
        else:
            prev_signal = None
            for signal in registers[0]:
                if total_signal is None:
                    prev_signal = signal
                    total_signal = np.array([np.zeros(len(signal[0])), np.zeros(len(signal[0]))])
                #print n_signals,
                temp_signal = registers[0].combine(signal, prev_signal,reader_position,x_to_xp = self.x_to_xp)
                if temp_signal[1] is not None:
                    total_signal = total_signal + temp_signal
                else:
                    total_signal[0] = total_signal[0] + temp_signal[0]
                n_signals += 1
                prev_signal = signal

        return_signal = total_signal[0]/float(n_signals)
        if self.x_to_xp == True:
            return_signal *= self.phase_conv_coeff

        return return_signal


# class RightAnglePickups(object):
#     # TODO: return tuplex of vectors (real and imag)
#     # Assumes that betatron phase difference between pickups is pi/2 (+ n*2 pi). Thus, the betatron amplitude is
#     # a quadratic sum of signals from the pickups independently of the betatron phase angle
#
#     def __init__(self, register_1,register_2):
#         self.register_1 = register_1
#         self.register_2 = register_2
#
#
#
#
#
#     def mix(self,registers,reader_position):
#         total_signal = None
#         n_signals = 0
#
#         theta = registers[1].position-registers[0].position
#         c = np.cos(theta)
#         s = np.sin(theta)
#
#         for signal_1, signal_2 in zip(registers[0],registers[1]):

