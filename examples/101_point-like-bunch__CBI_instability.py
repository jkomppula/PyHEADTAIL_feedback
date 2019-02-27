from __future__ import division
import sys, os
import sys
# PyHEADTAIL location if it's not already in the PYTHONPATH environment variable
sys.path.append('../../')
import matplotlib.pyplot as plt

import sys
import datetime
import numpy as np
from scipy.constants import c, e
from PyHEADTAIL_feedback.signal_tools.signal_generators import SimpleBeam, CircularPointBeam
from PyHEADTAIL_feedback.signal_tools.trackers_and_kickers import LinearWake, Damper, WakeSourceFromFile, ResistiveWallWakeSource
from PyHEADTAIL_feedback.signal_tools.trackers_and_kickers import Tracer, AvgValueTracer, DCSuppressor, Noise, track_beam
from PyHEADTAIL_feedback.processors.convolution import PhaseLinearizedLowpass, Gaussian, Sinc, Lowpass
from PyHEADTAIL_feedback.processors.register import TurnFIRFilter
from PyHEADTAIL_feedback.processors.convolution import Lowpass,  FIRFilter
from PyHEADTAIL_feedback.processors.resampling import DAC, HarmonicADC, BackToOriginalBins, Upsampler
from PyHEADTAIL_feedback.processors.resampling import Quantizer
from PyHEADTAIL_feedback.processors.addition import NoiseGenerator

def run(argv):
    np.random.seed(0)
    job_id = int(argv[0])
    case_id = 0

    intensity = 1.1e11
    damping_time = 20.
    
    it = job_id

    # SIMULATION PARAMETERS
    # ---------------------
    n_turns = 5000
    n_traces = 10

    # MACHINE PARAMETERS
    #-------------------
    circumference = 26658.883
    p0 = 450e9 * e / c
    accQ_x = 59.31
    
    beta_x = circumference / (2.*np.pi*accQ_x)
    n_bunches=3564
    
    filling_pattern_array = [
            # 3x[(72b + 8e) x 3 + 30 e]
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (30, 0),
            # 
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (30, 0),
            # 
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (30, 0),
            # 
            # 1e
            (1, 0),
            # 
            # [(72b + 8e) x 3 + 30 e + (72b + 8e) x 3 + 30 e + (72b + 8e) x 4 + 31 e]
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (30, 0),
            # 
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (30, 0),
            # 
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (31, 0),
            # 
            # 
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (30, 0),
            # 
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (30, 0),
            # 
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (31, 0),
            #
            #
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (30, 0),
            # 
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (30, 0),
            # 
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (31, 0),
            #
            #
            (80, 0),
            
            ]
    
    
#    filling_scheme = np.arange(n_bunches)
    filling_scheme = []
    counter = 0
    for s in filling_pattern_array:
        for i in range(s[0]):
            if s[1] == 1:
                filling_scheme.append(counter)
            counter += 1


    
    # WAKE SETTINGS
    # =============
    wakefile = 'wakeforhdtl_PyZbase_Allthemachine_450GeV_B1_LHC_inj_450GeV_B1.dat'
    n_turns_wake = 30

    
    wake_source= WakeSourceFromFile(wakefile,0,2)
    wake = LinearWake(wake_source, n_turns_wake)
 
    # GENERATES BEAM
    # --------------
    beam = CircularPointBeam(filling_scheme, circumference, n_bunches, intensity,
                             circular_overlapping=2, n_segments = 9, beta_x = beta_x)

    beam.set_beam_paramters(p0)
    beam.init_noise(1e-7)
    amplitude = 2e-6
    
    # Data
    # ---------------
    triggers = [
        ('turn', n_turns-n_traces),
        ('mean_abs_x', 0.3e-2)
    ]


    avg_data = AvgValueTracer(n_turns)
    bunch_by_bunch_data = Tracer(n_traces, variables=['x', 'xp'], triggers=triggers)

    trackers = [
            wake,
            avg_data,
            bunch_by_bunch_data
    ]
        
        
        
    # GENERATES OUTPUT DATA FILES
    # ---------------------------

    track_beam(beam, trackers, n_turns, accQ_x)

#    avg_data.save_to_file('case_' + str(it))
#    bunch_by_bunch_data.save_to_file('case_' + str(it))
    
#    %matplotlib inline
    import matplotlib.pyplot as plt

    n_points = 100
    m = avg_data.epsn_x
    m = (m>0)
    d = avg_data.mean_abs_x
    d = d[m]
    # print d 
    turns = np.linspace(1,len(d),len(d))
    coeffs_x = np.polyfit(turns[-n_points:], np.log(d[-n_points:]),1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax2.semilogy(turns, d)
    ax2.semilogy(turns, np.exp(coeffs_x[0]*turns + coeffs_x[1]),label='Growth time ' + str(int(1/coeffs_x[0])) + ' turns')
    ax2.set_ylim(0.5*np.min(d),2*np.max(d))    
    ax2.set_xlabel('Turn')
    ax2.set_ylabel('|Beam avg oscillation|')
    ax2.legend()

    n_plots = 6
    colors = [plt.cm.viridis(i) for i in np.linspace(0, 1, n_plots)]
    for i in range(n_plots):
        ax1.plot(bunch_by_bunch_data.x[i], '-', color=colors[i], label='Turn ' + str(int(len(turns)-n_traces-1+i)))

    ax1.set_xlabel('Bucket #')
    ax1.set_ylabel('BPM signal')
    ax1.legend()

    plt.show()

if __name__=="__main__":
	run(sys.argv[1:])
