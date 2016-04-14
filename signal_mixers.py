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
