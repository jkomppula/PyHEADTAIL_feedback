from __future__ import division

import sys, os
BIN = os.path.expanduser("../../../")
sys.path.append(BIN)

#import sys
import numpy as np
from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyHEADTAIL.impedances.wakes import WakeTable, WakeField
from PyHEADTAIL.monitors.monitors import BunchMonitor, SliceMonitor
from PyHEADTAIL.machines.synchrotron import Synchrotron
from PyHEADTAIL_feedback.processors.multiplication import ChargeWeighter
from PyHEADTAIL_feedback.processors.convolution import Gaussian
from PyHEADTAIL_feedback.processors.register import TurnDelay
from PyHEADTAIL_feedback.feedback import OneboxFeedback, PickUp, Kicker
from PyHEADTAIL_feedback.processors.register import TurnDelay, TurnFIRFilter, Register, FIRCombiner
from PyHEADTAIL_feedback.processors.misc import Average
from PyHEADTAIL_feedback.processors.misc import Bypass
from scipy.signal import hilbert
from PyHT_map_generator import generate_one_turn_map, find_object_locations

from matplotlib import gridspec


# This is a test, which tests how well the combiner roates an ideal signal. Horizontal values represent values from
# the combiner while the vectors from the origin represent values given to the register.

import matplotlib.pyplot as plt
import time
from scipy.constants import c, e, m_p


n_macroparticles = 100000
n_turns          = 1000

def analyse_data(d):
    # Calculates growth/damping time from the simulated data
    
    if np.mean(np.abs(d[:20])) <np.mean(np.abs(d[-20:])):
        stable = 0
        print('Growth')
        i_max = -50
        i_min = -250
        d= d[:int(1.2*i_max)]
    else:
        stable = 1
        print('Decay ')
        i_max = 250
        i_min = 50
    
    try:
        analytic_signal = hilbert(d)
        amplitude_envelope = np.abs(analytic_signal)

        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = (np.diff(instantaneous_phase)/(2.0*np.pi) * 1.)

        turns = np.linspace(1,len(d),len(d))

    #     try:
        tune = np.mean(instantaneous_frequency[i_min:i_max])
        coeffs = np.polyfit(turns[i_min:i_max], np.log(amplitude_envelope[i_min:i_max]), 1)
#        print('$\tau_d$=' + str(-1./coeffs[0]) + ' turns, Q=' + str(tune))
        return -1./coeffs[0], tune, stable
    except:
#        print('ERROR')
        tune = 0.
        coeffs = (1e-5,0)
        stable = -1
        amplitude_envelope = None
        instantaneous_frequency = None
        turns = None

        return -1./coeffs[0], tune, stable

def calculate_coefficients(Q, delay, additional_phase):
    # Generates a 3-tap FIR fiter coefficients for arbitrary 
    # phase advance and delay
    
    Q = -Q
    ppt = 2.* np.pi
    c12 = np.cos(Q *1.*ppt)
    s12 = np.sin(Q *1.*ppt)
    c13 = np.cos(Q *2.*ppt)
    s13 = np.sin(Q *2.*ppt)
    c14 = np.cos((Q *(2+delay)-additional_phase)*ppt)
    s14 = np.sin((Q *(2+delay)-additional_phase)*ppt)
    
    divider = -1.*(-c12*s13+c13*s12-s12+s13)

    cx1 = c14*(1-(c12*s13-c13*s12)/divider)+s14*(-c12+c13)/divider
    cx2 = (c14*(-(-s13))+s14*(-c13+1))/divider
    cx3 = (c14*(-(s12))+s14*(c12-1))/divider
    
    return np.array([cx3, cx2, cx1])


def find_objects_from_Twiss_file(optics_file, pickup_object, kicker_object,
                          phase_x_col, beta_x_col,
                          phase_y_col, beta_y_col):
    # Reads phases and betafunction values from Twiss file
    
    with open(optics_file) as f:
        content = f.readlines()
        content = [x.strip() for x in content]

    list_of_objects = [(pickup_object),(kicker_object)]
    object_locations = []
    for j, element in enumerate(list_of_objects):
        element_name = element
        line_found = False
        for i, l in enumerate(content):
            s = l.split()
            
            if s[0] == ('"' + element_name + '"'):
                line_found = True
                object_locations.append((element,
                                 float(s[phase_x_col]),
                                 float(s[beta_x_col]),
                                 float(s[phase_y_col]),
                                 float(s[beta_y_col])))
                                
        if line_found == False:
            raise ValueError('Element ' + element_name + ' not found from the optics file ' + optics_file)
    
    print('| Element name | Phase x | Beta X  | Phase y | Beta y  |')
    print('========================================================')
    for d in object_locations:
        print('| {:12.5} | {:7.3f} | {:6.2f}  | {:7.3f} | {:6.2f}  |'.format(d[0], d[1], d[2], d[3], d[4]))
    
    phase_difference_x = object_locations[1][1]-object_locations[0][1]
    phase_difference_y = object_locations[1][3]-object_locations[0][3]
    
    print('')
    print('')
    print('Phase advance difference between pickup ' + str(pickup_element) + ' and kicker ' + str(kicker_element) + ':')
    print('| Plane | Tune units | Radian  | Degrees |')
    print('|========================================|')
    print('| {:5.5} | {:10.3f} | {:6.3f}  | {:7.2f} |'.format('X', phase_difference_x, phase_difference_x*2.*np.pi, phase_difference_x*360.))
    print('| {:5.5} | {:10.3f} | {:6.3f}  | {:7.2f} |'.format('Y', phase_difference_y, phase_difference_y*2.*np.pi, phase_difference_y*360.))
    
    
    return phase_difference_x, phase_difference_y

def run(job_id, accQ_y, accQ_x, phase_filter_x, phase_filter_y, damping_time,
        total_filter_delay, optics_file, phase_x_col, beta_x_col, phase_y_col,
        beta_y_col, list_of_systems):
    # Main simulation file
    
#    job_id = 0
    chroma = 0
    intensity = 1e11
    gain = 2./damping_time
    charge = e
    mass = m_p
    
    alpha = 5.034**-2
    
    p0 = 300e9 * e / c
    Q_s = 0.02
    circumference = 160.
    s = None
    alpha_x = None
    alpha_y = None
    beta_x = circumference / (2.*np.pi*accQ_x)
    beta_y = circumference / (2.*np.pi*accQ_y)
    D_x = 0
    D_y = 0
    optics_mode = 'smooth'
    name = None
    n_segments = 1

    # detunings
    Qp_x = chroma
    Qp_y = chroma


    app_x = 0
    app_y = 0
    app_xy = 0

    longitudinal_mode = 'linear'

    h_RF = 2
    h_bunch = h_RF
    wrap_z = False

    machine = Synchrotron(
            optics_mode=optics_mode, circumference=circumference,
            n_segments=n_segments, s=s, name=name,
            alpha_x=alpha_x, beta_x=beta_x, D_x=D_x,
            alpha_y=alpha_y, beta_y=beta_y, D_y=D_y,
            accQ_x=accQ_x, accQ_y=accQ_y, Qp_x=Qp_x, Qp_y=Qp_y,
            app_x=app_x, app_y=app_y, app_xy=app_xy,
            alpha_mom_compaction=alpha, longitudinal_mode=longitudinal_mode,
            h_RF=np.atleast_1d(h_RF), p0=p0,
            charge=charge, mass=mass, wrap_z=wrap_z, Q_s=Q_s)
    
    beta_x = machine.transverse_map.beta_x[0]
    beta_y = machine.transverse_map.beta_y[0]


    # BEAM
    # ====
    epsn_x  = 300e-6
    epsn_y  = 300e-6
    sigma_z = 450e-9*c*machine.beta
    print('sigma_z: ' + str(sigma_z))
    bunch   = machine.generate_6D_Gaussian_bunch(
        n_macroparticles, intensity, epsn_x, epsn_y, sigma_z=sigma_z, filling_scheme = [0], matched=False)

    init_offset = 1e-2
    bunch.x = bunch.x + init_offset
    bunch.y = bunch.y + init_offset
    
    # CREATE BEAM SLICERS
    # ===================
    slicer_for_wakefields   = UniformBinSlicer(50, z_cuts=(-4*sigma_z, 4*sigma_z), circumference = circumference, h_bunch=h_bunch)
    
    
    # additional signal processing delay after one turn
    additional_filter_delay = total_filter_delay - 1
    
    Q_x            = accQ_x
    Q_y            = accQ_y
    
    pyHT_obj_list = []
    map_element_list = []
    
    # Total number of pickups and kickers, determines fractional gains for 
    # each pickup-kicker pair
    gain_divider = 1.
    
    for i, system in enumerate(list_of_systems):
        object_locations = find_object_locations(system, optics_file,
                                             phase_x_col, beta_x_col, phase_y_col, beta_y_col)
        
        plane = system[0][1]
        for j in range(len(system)-1):
            
            # phase advance difference between the pickup and kicker
            if plane == 'x':
                phase_advance = object_locations[0][1] - object_locations[j+1][1]
            elif plane == 'y':
                phase_advance = object_locations[0][3] - object_locations[j+1][3]
            else:
                raise ValueError('Unknown plane')
         
            # Checks if pickup is before or after the kickers and 
            # adjusts the register delay if needed
            if phase_advance >= 0.:
                delay = additional_filter_delay+1
            else:
                delay = additional_filter_delay
            
            # 90 deg extra phase advance for the x-readings to x' kicks -conversion
            phase_advance = phase_advance+0.25
            
            # Signal processing model for a kicker. In this case we have an ideal 
            # kicker.
            kicker_processors = [Bypass()]
            
            
            # Generates the pickup and kicker objects
            if plane == 'x':

                pickup_processors = [
                    ChargeWeighter(normalization = 'segment_average'),
                    Average(avg_type = 'total'),
                    Register(8, Q_x, delay)
                ]
                kicker_registers = [pickup_processors[-1]]
                combiner = (FIRCombiner(phase_filter_x, kicker_registers, 0., beta_x, beta_conversion = '90_deg'), None)
                pickup = PickUp(slicer_for_wakefields,
                         pickup_processors, None, 
                         object_locations[j+1][1], beta_x,
                         object_locations[j+1][3], beta_y)
                kicker = Kicker(gain/gain_divider, slicer_for_wakefields, 
                     kicker_processors, None,
                     kicker_registers, None,
                     object_locations[0][1], beta_x,
                     object_locations[0][3], beta_y, combiner=combiner)
    
            elif plane == 'y':
                pickup_processors = [
                    ChargeWeighter(normalization = 'segment_average'),
                    Average(avg_type = 'total'),
                    Register(8, Q_y, delay)
                ]
                kicker_registers = [pickup_processors[-1]]
                combiner = (None, FIRCombiner(phase_filter_y, kicker_registers, 0.,beta_y, beta_conversion = '90_deg'))
                pickup = PickUp(slicer_for_wakefields,
                         None, pickup_processors, 
                         object_locations[j+1][1], beta_x,
                         object_locations[j+1][3], beta_y)
                kicker = Kicker(gain/gain_divider, slicer_for_wakefields, 
                     None, kicker_processors,
                     None, kicker_registers,
                     object_locations[0][1], beta_x,
                     object_locations[0][3], beta_y, combiner=combiner)
            
            pyHT_obj_list.append(pickup)
            pyHT_obj_list.append(kicker)
            map_element_list.append(system[j+1])
            map_element_list.append(system[0])

    # generates a new one turn map where pickup and kicker are in different locations
    new_map = generate_one_turn_map(machine.one_turn_map,
                        map_element_list, pyHT_obj_list,
                        optics_file, phase_x_col, beta_x_col, phase_y_col, beta_y_col,
                        machine.circumference, alpha_x=None, beta_x=beta_x, D_x=0.,
                        alpha_y=None, beta_y=beta_y, D_y=0., accQ_x=accQ_x, 
                        accQ_y=accQ_y, Qp_x=chroma, Qp_y=chroma, app_x=0., 
                        app_y=0., app_xy=0., other_detuners=[], use_cython=False)
    machine.one_turn_map = new_map
    
    
    # tracks beam and returns turn-by-turn bunch positions
    output_data = np.zeros((n_turns,3))
    for i in range(n_turns):
        t0 = time.clock()
        output_data[i,0] = i
        output_data[i,1] = bunch.mean_x()
        output_data[i,2] = bunch.mean_y()

        machine.track(bunch)
        if (np.abs(bunch.mean_x()) > 4*init_offset) or (np.abs(bunch.mean_y()) > 4*init_offset):
            output_data = output_data[:(i+1),:]
            break

        if i%100 is not 0:
            continue

        print('{:4d} \t {:+3e} \t {:+3e} \t {:+3e} \t {:3e} \t {:3e} \t {:3f} \t {:3f} \t {:3f} \t {:3s}'.format(i, bunch.mean_x(), bunch.mean_y(), bunch.mean_z(), bunch.epsn_x(), bunch.epsn_y(), bunch.epsn_z(), bunch.sigma_z(), bunch.sigma_dp(), str(time.clock() - t0)))

    print('\n*** Successfully completed!')
    
    return output_data

###########################################
## READS PHASE ADVANCE FROM A TWISS FILE ##
###########################################

twiss_file = 'example_twiss.dat'
pickup_element = 'R4VM1'
kicker_element = 'R6VM1'

  
phase_x_column = 5
beta_x_column = 3
phase_y_column = 8
beta_y_column = 6



phase_difference_x, phase_difference_y = find_objects_from_Twiss_file(twiss_file, pickup_element, kicker_element,
                          phase_x_column, beta_x_column,
                          phase_y_column, beta_y_column)


########################################
## CALCULATES FIR FILTER COEFFICIENTS ##
########################################

# Tune 
Q_filter = 3.75 
Q_x = 4.28

# Minimum signal processing delay in turns
delay = 3
damping_time = 50.

# Phase advance between the pickup and the kicker. Additional 0.25 must
# be added because the pickup reads beam displacement but changes beam
# angle.
additional_phase = phase_difference_y + 0.25

phase_filter_y = calculate_coefficients(Q_filter, delay, additional_phase)
phase_filter_x = [0.,0.,0.]
    

######################
## RUNS SIMULATIONS ##
######################

list_of_systems = [
        [(kicker_element, 'y'),(pickup_element, 'y')],
        ]

tunes = np.linspace(3.6,3.9,41)
#tunes = np.linspace(3.7,3.8,7)
damping_times = np.zeros((len(tunes),2))

    
for job_id, accQ_y in enumerate(tunes):
    print('#########################')
    print('Job ' + str(job_id+1) + '/' + str(len(tunes)))
    print('#########################')
    
    simulation_data = run(job_id, accQ_y, Q_x, phase_filter_x, phase_filter_y, damping_time,
            delay, twiss_file, phase_x_column, beta_x_column, phase_y_column,
            beta_y_column, list_of_systems)
    
    t_d, tune, stable = analyse_data(simulation_data[:,2])

    damping_times[job_id,0] = accQ_y
    damping_times[job_id,1] = t_d

np.savetxt('FIR_filter_tune_acceptance.txt', damping_times)

#########################
## PLOTS DAMPING TIMES ##
#########################

print(damping_times)

Q_s_data = np.loadtxt('Q_s_data.csv')
Q_data = np.loadtxt('Tune_data.csv')


# selects valid data points which are between min and max damping time
min_damping_time = 10.
max_damping_time = 200.
damping_times = damping_times[(damping_times[:,1]>min_damping_time)*(damping_times[:,1]<max_damping_time),:]


# Plots betatrone tune and side bands
fig = plt.figure(figsize=(8, 4)) 
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1.5]) 
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1], sharey=ax1)

x=np.linspace(0,10,1000)

Q_y = np.interp(x, Q_data[:,0], Q_data[:,1]) 
Q_s = np.interp(x, Q_s_data[:,0], Q_s_data[:,1]) 

ax1.plot(x,Q_y,'k-', label=r'$Q_y$')
ax1.plot(x,Q_y+Q_s, 'r-', label=r'$Q_y\pm Q_s$')
ax1.plot(x,Q_y-Q_s, 'r-')
ax1.plot(x,Q_y+2*Q_s, 'r--', label=r'$Q_y\pm 2Q_s$')
ax1.plot(x,Q_y-2*Q_s, 'r--')
ax1.plot(x,Q_y+3*Q_s, 'r-.', label=r'$Q_y\pm 3Q_s$')
ax1.plot(x,Q_y-3*Q_s, 'r-.')
ax1.plot(x,Q_y+4*Q_s, 'r:', label=r'$Q_y\pm 4Q_s$')
ax1.plot(x,Q_y-4*Q_s, 'r:')

ax1.legend()
ax1.set_xlim(0,10)
ax1.set_xlabel('Time [ms]')
ax2.set_xlabel('Damping time [turns]')
ax1.set_ylabel('Tune')
ax1.set_ylim(3.6,3.9)
ax2.set_ylim(3.6,3.9)

# Plots green are for the tune acceptance
y_min = np.min(damping_times[:,0])
y_max = np.max(damping_times[:,0])
ax1.fill_between([0,1e3], y_min, y_max, facecolor='green', alpha=0.15)
ax2.fill_between([0,1e3], y_min, y_max, facecolor='green', alpha=0.15)


# Plots damping times and saves the data
ax2.set_xlim(20, 200)
#fig.suptitle('Optimized filters for specific tunes')
ax2.plot(damping_times[:,1],damping_times[:,0],'C2.-', label='$Q_{design}$=' + str(Q_filter) + '\nDelay ' + str(delay) + ' turns')
ax2.legend(loc='top left', bbox_to_anchor=(1, 1.0))
plt.tight_layout()
fig.subplots_adjust(top=0.90)

#fig.savefig('tune_acceptance_vs_ramp__filter.png', format='png', dpi=300)
plt.show()