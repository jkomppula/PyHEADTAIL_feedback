# run this file by using command:
#$ mpirun -np 4 python FillingScheme_and_Feedback_Test.py


from __future__ import division

import sys, os
BIN = os.path.expanduser("../../")
sys.path.append(BIN)

import time
import numpy as np
import seaborn as sns
from mpi4py import MPI
import matplotlib.pyplot as plt
from scipy.constants import c, e, m_p, pi

from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyHEADTAIL_feedback.feedback import Kicker, PickUp
from PyHEADTAIL_feedback.processors.multiplication import ChargeWeighter
from PyHEADTAIL_feedback.processors.convolution import Sinc
from PyHEADTAIL_feedback.processors.misc import Bypass
from PyHEADTAIL_feedback.processors.register import HilbertPhaseShiftRegister
from PyHEADTAIL_feedback.processors.signal import BeamParameters
from PyHEADTAIL_feedback.processors.resampling import ADC,DAC,UpSampler


def pick_signals(processor, source = 'input'):

    if source == 'input':
        bin_edges = processor.input_signal_parameters.bin_edges
        raw_signal = processor.input_signal
    elif source == 'output':
        bin_edges = processor.output_signal_parameters.bin_edges
        raw_signal = processor.output_signal
    else:
        raise ValueError('Unknown value for the data source')

    print 'len(bin_edges): ' + str(len(bin_edges))
    print 'len(raw_signal): ' + str(len(raw_signal))

    t = np.zeros(len(raw_signal)*4)
    z = np.zeros(len(raw_signal)*4)
    bins = np.zeros(len(raw_signal)*4)
    signal = np.zeros(len(raw_signal)*4)
    value = 1.


    for i, edges in enumerate(bin_edges):
        z[4*i] = edges[0]
        z[4*i+1] = edges[0]
        z[4*i+2] = edges[1]
        z[4*i+3] = edges[1]
        bins[4*i] = 0.
        bins[4*i+1] = value
        bins[4*i+2] = value
        bins[4*i+3] = 0.
        signal[4*i] = 0.
        signal[4*i+1] = raw_signal[i]
        signal[4*i+2] = raw_signal[i]
        signal[4*i+3] = 0.
        value *= -1

    t = z/c

    return (t, z, bins, signal)


def kicker(bunch):
    bunch.x *= 0
    bunch.xp *= 0
    bunch.y *= 0
    bunch.yp *= 0
    bunch.x[:] += 2e-2 * np.sin(2.*pi*np.mean(bunch.z)/1000.)

plt.switch_backend('TkAgg')
sns.set_context('talk', font_scale=1.3)
sns.set_style('darkgrid', {
    'axes.edgecolor': 'black',
    'axes.linewidth': 2,
    'lines.markeredgewidth': 1})


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

n_turns = 100
chroma = 0

n_segments = 1
n_bunches = 13
filling_scheme = [401 + 20*i for i in range(n_bunches)]
n_macroparticles = 40000
intensity = 2.3e11


# BEAM AND MACHNINE PARAMETERS
# ============================
from test_tools import MultibunchMachine
machine = MultibunchMachine(n_segments=n_segments)

epsn_x = 2.e-6
epsn_y = 2.e-6
sigma_z = 0.081

bunches = machine.generate_6D_Gaussian_bunch_matched(
    n_macroparticles, intensity, epsn_x, epsn_y, sigma_z=sigma_z,
    filling_scheme=filling_scheme, kicker=kicker)

# CREATE BEAM SLICERS
# ===================
slicer = UniformBinSlicer(50, n_sigma_z=3)

delay = 1
n_values = 3

bunch_spacing = 2.49507468767912e-08
f_ADC = 1./(bunch_spacing)
signal_length = 2.*bunch_spacing

f_c = 50e6

pickup_processors_x = [
    ChargeWeighter(normalization = 'average',store_signal  = True),
    ADC(f_ADC, n_bits = 8, input_range = (-1e-3,1e-3), signal_length = signal_length,store_signal  = True),
    HilbertPhaseShiftRegister(n_values, machine.accQ_x, delay,store_signal  = True)
]
pickup_processors_y = [
    ChargeWeighter(normalization = 'average',store_signal  = True),
    ADC(f_ADC, n_bits = 8, input_range = (-1e-3,1e-3), signal_length = signal_length,store_signal  = True),
    HilbertPhaseShiftRegister(n_values, machine.accQ_x, delay,store_signal  = True)
]


pickup_beam_parameters_x = BeamParameters(1.*2.*pi/float(n_segments)*machine.accQ_x,machine.beta_x_inj)
pickup_beam_parameters_y = BeamParameters(1.*2.*pi/float(n_segments)*machine.accQ_y,machine.beta_y_inj)

pickup_map = PickUp(slicer,pickup_processors_x,pickup_processors_y,
       pickup_beam_parameters_x, pickup_beam_parameters_y, mpi = True)

registers_x = [pickup_processors_x[2]]
registers_y = [pickup_processors_y[2]]

kicker_processors_x = [
    UpSampler(3,kernel=[0,1,0],store_signal  = True),
    Sinc(1*f_c,store_signal  = True),
    DAC(store_signal  = True)
]
kicker_processors_y = [
    UpSampler(3,kernel=[0,1,0],store_signal  = True),
    Sinc(1*f_c,store_signal  = True),
    DAC(store_signal  = True)
]

kicker_beam_parameters_x = BeamParameters(2.*2.*pi/float(n_segments)*machine.accQ_x,machine.beta_x_inj)
kicker_beam_parameters_y = BeamParameters(2.*2.*pi/float(n_segments)*machine.accQ_y,machine.beta_y_inj)

gain = 0.1

kicker_map = Kicker(gain, slicer, kicker_processors_x, kicker_processors_y,
                    registers_x, registers_y, kicker_beam_parameters_x, kicker_beam_parameters_y, mpi = True)

# TRACKING LOOP
# =============

new_one_turn_map = []
for i, m in enumerate(machine.one_turn_map):

    if i == 1:
        new_one_turn_map.append(pickup_map)

    if i == 2:
        new_one_turn_map.append(kicker_map)

    new_one_turn_map.append(m)

machine.one_turn_map = new_one_turn_map

s_cnt = 0
monitorswitch = False
if rank == 0:
    print '\n--> Begin tracking...\n'

print 'Tracking'
for i in range(n_turns):

    if rank == 0:
        t0 = time.clock()
    machine.track(bunches)

    if rank == 0:
        t1 = time.clock()
        print('Turn {:d}, {:g} ms, {:s}'.format(i, (t1-t0)*1e3, time.strftime(
            "%d/%m/%Y %H:%M:%S", time.localtime())))

if rank == 0:
    fig, (ax1, ax2) = plt.subplots(2, figsize=(14, 14), sharex=False)
    fig.suptitle('Pickup processors', fontsize=20)
    for i, processor in enumerate(pickup_processors_x):
        print 'Processor: ' + str(i)
        t, z, bins, signal = pick_signals(processor,'output')
        ax1.plot(z, bins*(1.-0.1*i), label =  processor.label)
        ax2.plot(z, signal, label =  processor.label)
        if i == 0:
            print t
            print z
            print bins
            print signal

    # ax3.plot(processors_x[0]._CDF_time,processors_x[0]._PDF,'r-')
    ax1.set_ylim([-1.1, 1.1])
    ax1.set_xlabel('Z position [m]')
    ax1.set_ylabel('Bin set')
    ax1.legend(loc='upper left')



    ax2.set_xlabel('Z position [m]')
    ax2.set_ylabel('Signal')
    ax2.legend(loc='upper left')


    fig, (ax3, ax4) = plt.subplots(2, figsize=(14, 14), sharex=False)
    fig.suptitle('Kicker processors', fontsize=20)


    for i, processor in enumerate(kicker_processors_x):
        print 'Processor: ' + str(i)
        t, z, bins, signal = pick_signals(processor,'output')
        ax3.plot(z, bins*(1.-0.1*i), label =  processor.label)
        ax4.plot(z, signal, label =  processor.label)
        if i == 0:
            print t
            print z
            print bins
            print signal

    # ax3.plot(processors_x[0]._CDF_time,processors_x[0]._PDF,'r-')
    ax3.set_ylim([-1.1, 1.1])
    ax3.set_xlabel('Z position [m]')
    ax3.set_ylabel('Bin set')
    ax3.legend(loc='upper left')

    ax4.set_xlabel('Z position [m]')
    ax4.set_ylabel('Signal')
    ax4.legend(loc='upper left')

    plt.show()
