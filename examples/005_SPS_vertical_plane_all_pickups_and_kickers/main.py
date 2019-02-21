from __future__ import division

import sys, os
#BIN = os.path.expanduser("/home/jani/Programming/DevClones/PyHEADTAIL_feedback__multi_pickup_map_generator/")
#sys.path.append(BIN)

#import sys
import numpy as np
from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyHEADTAIL.impedances.wakes import WakeTable, WakeField
from PyHEADTAIL.feedback.transverse_damper import TransverseDamper
from PyHEADTAIL.monitors.monitors import BunchMonitor, SliceMonitor
from PyHEADTAIL.machines.synchrotron import Synchrotron
from PyHEADTAIL_feedback.processors.multiplication import ChargeWeighter
from PyHEADTAIL_feedback.processors.convolution import Gaussian
from PyHEADTAIL_feedback.processors.register import TurnDelay
from PyHEADTAIL_feedback.feedback import OneboxFeedback, PickUp, Kicker
from PyHEADTAIL_feedback.processors.register import TurnDelay, TurnFIRFilter, Register, FIRCombiner
from PyHEADTAIL_feedback.processors.misc import Average
from PyHEADTAIL_feedback.processors.misc import Bypass
from MD4063_filter_functions import calculate_coefficients_3_tap, calculate_hilbert_notch_coefficients
from PyHT_map_generator import generate_one_turn_map, find_object_locations

# This is a test, which tests how well the combiner roates an ideal signal. Horizontal values represent values from
# the combiner while the vectors from the origin represent values given to the register.


import time
from scipy.constants import c, e, m_p


n_macroparticles = 100000
#n_turns          = 1000
n_turns          = 2000
wakefile1        = './FCC_wakes_2018_01_3p3_TeV.dat'
outputpath = 'Data'


# it = 1
# intensity = 1e11
# chroma = 0
def run(job_id, tune_error, case_idx, damping_time):
#    job_id = 0
    chroma = 0
    it = job_id
    intensity = 1e11
#    h_bunch = 13068

#    tune_errors = np.linspace(-0.04,0.04,33)
#    damping_times = np.logspace(np.log10(2),np.log10(200),33)
#    n_jobs_per_case = len(tune_errors)*len(damping_times)
#    case_idx = int(job_id/(n_jobs_per_case))
#    
#    settings_idx = int(job_id%n_jobs_per_case)
#    tune_error_idx = int(settings_idx%len(tune_errors))
#    gain_idx = int(settings_idx/len(tune_errors))
#    
#    norm_coeff = 1.0
#    gain = norm_coeff * 2./float(damping_times[gain_idx])
#    
#
#    tune_error = tune_errors[tune_error_idx]
    
#    case_idx = 1
#    tune_error = 0
    gain = 2./damping_time
#    if  case_idx == 0:
#        gain = gain*0.98

#def run(intensity, chroma, damping_rate = 1e6):


    # BEAM AND MACHNINE PARAMETERS
    # ============================
    # BEAM AND MACHNINE PARAMETERS
    # ============================
    charge = e
    mass = m_p
    alpha = 0.00308

    p0 = 26e9 * e / c
    

    accQ_x_filter = 20.13
    accQ_y_filter = 20.18
    accQ_x = accQ_x_filter+tune_error
    accQ_y = accQ_y_filter+tune_error
    Q_s = 0.017
    circumference = 1100*2*np.pi
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

#    h_bunch = h_RF
    h_RF = 4620
    h_bunch = 462
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
    epsn_x  = 2.2e-6
    epsn_y  = 2.2e-6
    sigma_z = 0.08
    bunch   = machine.generate_6D_Gaussian_bunch(
        n_macroparticles, intensity, epsn_x, epsn_y, sigma_z=sigma_z)
    bunch.x = bunch.x + 1e-3
    bunch.y = bunch.y + 1e-3
    epsn_x_0 = bunch.epsn_x()
    epsn_y_0 = bunch.epsn_y()

    # CREATE BEAM SLICERS
    # ===================
    slicer_for_slicemonitor = UniformBinSlicer(50, z_cuts=(-3*sigma_z, 3*sigma_z))
    slicer_for_wakefields   = UniformBinSlicer(500, z_cuts=(-4*sigma_z, 4*sigma_z))


    # CREATE WAKES
    # ============
    # FIXME? is this correct?
    # FIXME? Zero line added to the wake file
#    wake_table1          = WakeTable(wakefile1,
#                                     ['time', 'dipole_x', 'dipole_y'])
#    wake_field           = WakeField(slicer_for_wakefields, wake_table1)
#
    
    additional_filter_delay = 0.    

    optics_file = 'SPS-Q20-2015v1.tfs'
    
    phase_x_col = 12
    beta_x_col = 9
    phase_y_col = 17
    beta_y_col = 14
    
    Q_x            = accQ_x_filter
    Q_y            = accQ_y_filter
    
    list_of_systems = [
            [('BDH.21437', 'x'),('BPCR.21459', 'x'),('BPCR.22172', 'x')],
            [('BDH.21451', 'x'),('BPCR.21459', 'x'),('BPCR.22172', 'x')],
            [('BDV.21455', 'y'),('BPCR.21459', 'y'),('BPCR.22172', 'y')],
            [('BDV.22176', 'y'),('BPCR.21459', 'y'),('BPCR.22172', 'y')],
            ]
    
    pyHT_obj_list = []
    map_element_list = []
    
    gain_divider = 4.
    
    for i, system in enumerate(list_of_systems):
        object_locations = find_object_locations(system, optics_file,
                                             phase_x_col, beta_x_col, phase_y_col, beta_y_col)
        plane = system[0][1]
        for j in range(len(system)-1):
            if plane == 'x':
                phase_advance = object_locations[0][1] - object_locations[j+1][1]
            elif plane == 'y':
                phase_advance = object_locations[0][3] - object_locations[j+1][3]
            else:
                raise ValueError('Unknown plane')
            
            if phase_advance >= 0.:
                delay = additional_filter_delay+1
            else:
                delay = additional_filter_delay
                
            phase_advance = phase_advance+0.25
            
            kicker_processors = [Bypass()]
            
            if plane == 'x':
                if case_idx == 0:
                    phase_filter = calculate_coefficients_3_tap(Q_x, 1+additional_filter_delay, phase_advance)
                elif case_idx == 1:
                    phase_filter = calculate_hilbert_notch_coefficients(Q_x, 1+additional_filter_delay, phase_advance)
                else:
                    raise ValueError('Unknown case idx')
                pickup_processors = [
                    ChargeWeighter(normalization = 'segment_average'),
                    Average(avg_type = 'total'),
                    Register(8, Q_x, delay)
                ]
                kicker_registers = [pickup_processors[-1]]
                combiner = (FIRCombiner(phase_filter, kicker_registers, 0., beta_x, beta_conversion = '90_deg'), None)
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
                if case_idx == 0:
                    phase_filter = calculate_coefficients_3_tap(Q_y, 1+additional_filter_delay, phase_advance)
                elif case_idx == 1:
                    phase_filter = calculate_hilbert_notch_coefficients(Q_y, 1+additional_filter_delay, phase_advance)
                else:
                    raise ValueError('Unknown case idx')
                pickup_processors = [
                    ChargeWeighter(normalization = 'segment_average'),
                    Average(avg_type = 'total'),
                    Register(8, Q_y, delay)
                ]
                kicker_registers = [pickup_processors[-1]]
                combiner = (None, FIRCombiner(phase_filter, kicker_registers, 0.,beta_y, beta_conversion = '90_deg'))
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
            
    print(map_element_list)
    print(pyHT_obj_list)
    new_map = generate_one_turn_map(machine.one_turn_map,
                        map_element_list, pyHT_obj_list,
                        optics_file, phase_x_col, beta_x_col, phase_y_col, beta_y_col,
                        machine.circumference, alpha_x=None, beta_x=beta_x, D_x=0.,
                        alpha_y=None, beta_y=beta_y, D_y=0., accQ_x=accQ_x, 
                        accQ_y=accQ_y, Qp_x=chroma, Qp_y=chroma, app_x=0., 
                        app_y=0., app_xy=0., other_detuners=[], use_cython=False)
    print('NEW MAP:')
    print(new_map)
    machine.one_turn_map = new_map
    
    print('object_locations:')
    print(object_locations)

    
    output_data = np.zeros((n_turns,3))
#    turns = np.zeros(n_turns)
#    mean_x = np.zeros(n_turns)
#    mean_y = np.zeros(n_turns)

    for i in range(n_turns):
        t0 = time.clock()
        output_data[i,0] = i
        output_data[i,1] = bunch.mean_x()
        output_data[i,2] = bunch.mean_y()

        machine.track(bunch)

        if i%10 is not 0:
            continue

        print('{:4d} \t {:+3e} \t {:+3e} \t {:+3e} \t {:3e} \t {:3e} \t {:3f} \t {:3f} \t {:3f} \t {:3s}'.format(i, bunch.mean_x(), bunch.mean_y(), bunch.mean_z(), bunch.epsn_x(), bunch.epsn_y(), bunch.epsn_z(), bunch.sigma_z(), bunch.sigma_dp(), str(time.clock() - t0)))

    print('\n*** Successfully completed!')
    
    return output_data
    

if __name__=="__main__":
    
    arguments = sys.argv[1:]
    
    job_id = int(arguments[0])
    
    tune_errors = np.linspace(-0.08,0.08,33)
    damping_times = np.linspace(2,60,59)
    
    tune_idx = job_id%len(tune_errors)
    damping_idx = int(job_id/len(tune_errors))
    
    dQ = tune_errors[tune_idx]
    damping_time = damping_times[damping_idx]
    
    data_0 = run(job_id, dQ, 0, damping_time)
    data_1 = run(job_id, dQ, 1, damping_time)
    
    n_columns = len(data_0[0,:])+len(data_0[1,:])
    n_rows = len(data_0[:,0])
    
    out_data = np.zeros((n_rows, n_columns))
    
    for i in range(len(data_0[0,:])):
        out_data[:,i] = data_0[:,i]
        
    for i in range(len(data_1[0,:])):
        out_data[:,i+len(data_0[0,:])] = data_1[:,i]
        
    np.savetxt('Data/job_' + str(job_id) + '.dat', out_data)
        
    

