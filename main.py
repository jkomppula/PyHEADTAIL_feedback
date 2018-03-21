from __future__ import division
import sys
import datetime
import numpy as np
from scipy.constants import c, e

import sys, os
BIN = os.path.expanduser("../")
sys.path.append(BIN)
from PyHEADTAIL_feedback.signal_tools.signal_generators import SimpleBeam, CircularPointBeam
from PyHEADTAIL_feedback.signal_tools.trackers_and_kickers import track_beam, LinearWake, Damper, WakeSourceFromFile, BeamRotator,IdealDamper
from PyHEADTAIL_feedback.signal_tools.trackers_and_kickers import Tracer, AvgValueTracer, DCSuppressor, CircularComplexWake
from PyHEADTAIL_feedback.processors.convolution import PhaseLinearizedLowpass, Gaussian, Sinc, Lowpass

def my_kick_profile(amplitude, risetime, flat, falltime):
    
    def kick_amplitude(turn):
        if turn < risetime:
            return amplitude*turn/float(risetime)
        
        elif (turn >= risetime) and (turn < (risetime+flat)) :
            return amplitude
        
        elif (turn >= (risetime+flat)) and (turn < (risetime+flat+falltime)) :
            return amplitude - amplitude*(turn-(risetime+flat))/float(falltime)
        
        else:
            return 0
    return kick_amplitude

class MyKicker(object):
    """ A tracker, which kicks beam after a given number of turns.
    """

    def __init__(self, tune, amp_function, kicked_bunches, start_from=0, kick_var='x'):
        """
        Parameters
        ----------
        kick_function : function
            A python function, which calculates the kick. The input parameter for the function
            is the beam property (Numpy array) determined by the parameter seed_var and the
            function returns a Numpy array which is added to the beam property determined in the
            parameter kick_var
        kick_turns : int or list
            A turn, when the kick is applied. If a list of numbers is given, the kick is applied
            every turn given in the list.
        kick_var : str
            A beam property which is changed by the kick, e.g. 'x', 'xp', 'y', 'yp', etc
        seed_var : str
            A beam property which is used as a seed for the kick function,
            e.g. 'z', 'x', 'xp', etc
        """

        self._tune = tune
        self._amp_function = amp_function

        self.kicked_bunches = np.array(kicked_bunches)
        self._kick_var = kick_var
        self._start_from = start_from

        self._turn_counter = 0
        
        self.rotation_done = False

    def operate(self, beam, **kwargs):
        if self._turn_counter > self._start_from:
            turn = self._turn_counter - self._start_from
            new_vals = np.array(getattr(beam, self._kick_var))
#            print self.kicked_bunches
#            print self._amp_function
            new_vals[self.kicked_bunches] = new_vals[self.kicked_bunches] + self._amp_function(turn)*np.cos(2.*np.pi*turn*self._tune)          
            
            setattr(beam, self._kick_var, new_vals)

        self._turn_counter += 1

    @property
    def done(self):
        return False


def run(argv):
    job_id = int(argv[0])

    case_id = 0
    
    if case_id == 0:
        plane_id = 'x'
        method_id = 'linear'
        
    elif case_id == 1:
        plane_id = 'y'
        method_id = 'linear'
    else:
        raise ValueError('Unknown case_id')
        
        
    intensity = 1e11

    it = job_id

    # SIMULATION PARAMETERS
    # ---------------------
    n_turns = 4096
    n_turns = 700    # MACHINE PARAMETERS
    #-------------------
    circumference = 26658.883
    p0 = 6.5e12 * e / c
    
    if plane_id == 'x':
        accQ_x = 64.28
    elif plane_id == 'y':
        accQ_x = 65.31
        
    beta_x = circumference / (2.*np.pi*accQ_x)
    n_bunches=3564
#        bunch_spacing=25e-9
    #    intensity=1e11

    # FILLING SCHEME
    # --------------
    filling_scheme = []
    n_fills = 12
    fill_gap = 39
    batch_gap = 8
    bunches_per_batch = 72
    batches_per_fill = 3

    batch_length = bunches_per_batch+batch_gap
    fill_length = fill_gap + batches_per_fill*bunches_per_batch + (batches_per_fill - 1) * batch_gap

    for i in range(n_fills):
        for j in range(batches_per_fill):
            for k in range(bunches_per_batch):
                idx = i*fill_length + j*batch_length + k
                filling_scheme.append(idx)


    # GENERATES BEAM
    # --------------
    beam = CircularPointBeam(filling_scheme, circumference, n_bunches, intensity,
                             circular_overlapping=2, n_segments = 36, beta_x = beta_x, Q_x=accQ_x)

    beam.set_beam_paramters(p0)
    beam.init_noise(1e-7)

    # Data
    # ---------------
    triggers = [
        ('turn', 0)
    ]


    bunch_by_bunch_data = Tracer(n_turns, variables=['x'], triggers=triggers)
    bunch_by_bunch_data1 = Tracer(n_turns, variables=['x'], triggers=triggers)
    bunch_by_bunch_data2 = Tracer(n_turns, variables=['x'], triggers=triggers)
    bunch_by_bunch_data3 = Tracer(n_turns, variables=['x'], triggers=triggers)


#    # DAMPER SETTINGS
#    # ---------------
#
#    processors = [
#    #    GaussianLowpass(fc)
#        Gaussian(fc)
#    #    Lowpass(fc, normalization=('bunch_by_bunch', bunch_spacing))
#    ]
    damper = IdealDamper(2./40.0)
    
    amp_function = my_kick_profile(1e-4, 100, 100, 100)
    
    kicker = MyKicker(accQ_x, amp_function, [1])
    trackers = [
        kicker,
        damper,  
        bunch_by_bunch_data,
        BeamRotator(1,accQ_x/4.),
        bunch_by_bunch_data1,
        BeamRotator(1,accQ_x/4.),
        bunch_by_bunch_data2,
        BeamRotator(1,accQ_x/4.),
        bunch_by_bunch_data3,
        BeamRotator(1,accQ_x/4.),
    ]
    # GENERATES OUTPUT DATA FILES
    # ---------------------------
    track_beam(beam,trackers,n_turns,accQ_x)

    bunch_by_bunch_data.save_to_file('./case_' + str(it))
    import matplotlib.pyplot as plt
    f, ax = plt.subplots()
    ax.plot(bunch_by_bunch_data1.x[:,1])
    ax.set_xlim(0,700)
    plt.show()

if __name__=="__main__":
	run(sys.argv[1:])
