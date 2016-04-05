import numpy as np
import itertools
import math



# TODO: Check vector sum of complex numbers
class Averager(object):
    """The simplest possible signal mixer of pick ups, which calculates a phase weighted average of
    the pick up signals"""
    def __init__(self,channel,location_phase_angle):
        self.channel = channel
        self.location_phase_angle = location_phase_angle

    def mix(self,pickups):
        signal = None

        for index, pickup in enumerate(pickups):
            if signal is None:
                temp = pickup.sig_x(1)
                signal = np.zeros(len(temp))

            if self.channel == 'x':
                signal += pickup.sig_x(self.location_phase_angle)/len(pickups)
            elif self.channel == 'y':
                signal += pickup.sig_y(self.location_phase_angle)/len(pickups)

        return  signal
