'''
Simulation of beam interaction with coupling impedance and damper
for a single bunch
'''

# using Python 2.7:
from __future__ import division, print_function
range_ = range
range = xrange

import sys
# PyHEADTAIL location if it's not already in the PYTHONPATH environment variable
sys.path.append('../../')



import time, os

import numpy as np
import h5py
from scipy.constants import e, m_p, c

from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyHEADTAIL.impedances.wakes import WakeTable, WakeField
from PyHEADTAIL.feedback.feedback import IdealBunchFeedback
from PyHEADTAIL.mpi.mpi_data import my_rank
from PyHEADTAIL.monitors.monitors import (
    BunchMonitor, ParticleMonitor, SliceMonitor)
np.random.seed(my_rank())


n_macroparticles = 1000 # number of macro-particles to resolve the beam
n_turns = 1000 # simulation time
n_turns_slicemon = 64 # recording span of the slice statistics monitor

filling_scheme = []
n_batches = 4
batch_length = 72
gap = 8

for i in range(n_batches):
    for j in range(batch_length):
        filling_scheme.append(i*(batch_length+gap)+j)

# COMMENT THE NON-WANTED SET-UP:

# # injection
machine_configuration = 'LHC-injection'
wakefile = ('./wakeforhdtl_PyZbase_Allthemachine_450GeV'
            '_B1_LHC_inj_450GeV_B1.dat')

# flat-top
# machine_configuration = 'LHC_6.5TeV_collision_2016'
# wakefile = ('./wakeforhdtl_PyZbase_Allthemachine_6p5TeV'
#             '_B1_LHC_ft_6.5TeV_B1.dat')

# ---

def get_nonlinear_params(chroma, i_oct, p0=6.5e12*e/c):
    '''Arguments:
        - chroma: first-order chromaticity Q'_{x,y}, identical
          for both transverse planes
        - i_oct: octupole current in A (positive i_oct means
          LOF = i_oct > 0 and LOD = -i_oct < 0)
    '''
    # factor 2p0 is PyHEADTAIL's convention for d/dJx instead of
    # MAD-X's convention of d/d(2Jx)
    print('p0: ' + str(p0/(e/c)))
    
    app_x = 2 * p0 * 27380.10941 * i_oct / 100.
    app_y = 2 * p0 * 28875.03442 * i_oct / 100.
    app_xy = 2 * p0 * -21766.48714 * i_oct / 100.
    Qpp_x = 4889.00298 * i_oct / 100.
    Qpp_y = -2323.147896 * i_oct / 100.
    return {
        'app_x': app_x,
        'app_y': app_y,
        'app_xy': app_xy,
        'Qp_x': [chroma,],# Qpp_x],
        'Qp_y': [chroma,],# Qpp_y],
        # second-order chroma commented out above!
    }


def run(intensity, chroma=0, i_oct=0):
    '''Arguments:
        - intensity: integer number of charges in beam
        - chroma: first-order chromaticity Q'_{x,y}, identical
          for both transverse planes
        - i_oct: octupole current in A (positive i_oct means
          LOF = i_oct > 0 and LOD = -i_oct < 0)
    '''


    # BEAM AND MACHINE PARAMETERS
    # ============================
    from LHC import LHC
    # energy set above will enter get_nonlinear_params p0
    assert machine_configuration == 'LHC-injection'
    machine = LHC(n_segments=1,
                  machine_configuration=machine_configuration,
                  **get_nonlinear_params(chroma=chroma, i_oct=i_oct,p0=0.45e12*e/c))


    # BEAM
    # ====
    
    #print(filling_scheme)
    
    
    epsn_x = 3.e-6 # normalised horizontal emittance
    epsn_y = 3.e-6 # normalised vertical emittance
    sigma_z = 1.2e-9 * machine.beta*c/4. # RMS bunch length in meters

    beam = machine.generate_6D_Gaussian_bunch(
        n_macroparticles, intensity, epsn_x, epsn_y, sigma_z=sigma_z, 
        matched=True, filling_scheme=filling_scheme)
    
    bunch_list = beam.split_to_views()
    bunch = bunch_list[0]

    print ("\n--> Bunch length and emittance: {:g} m, {:g} eVs.".format(
            bunch.sigma_z(), bunch.epsn_z()))


    # CREATE BEAM SLICERS
    # ===================
    slicer_for_slicemonitor = UniformBinSlicer(
        50, z_cuts=(-3*sigma_z, 3*sigma_z))
    slicer_for_wakefields = UniformBinSlicer(
        50, z_cuts=(-3*sigma_z, 3*sigma_z),
        circumference=machine.circumference,
        h_bunch=machine.h_bunch)

    # CREATE WAKES
    # ============
    wake_table1 = WakeTable(wakefile,
                            ['time', 'dipole_x', 'dipole_y',
                             # 'quadrupole_x', 'quadrupole_y',
                             'noquadrupole_x', 'noquadrupole_y',
                             # 'dipole_xy', 'dipole_yx',
                             'nodipole_xy', 'nodipole_yx',
                            ])
    wake_field = WakeField(slicer_for_wakefields, wake_table1, mpi='linear_mpi_full_ring_fft')


    # CREATE DAMPER
    # =============
    #dampingtime = 50.
    #gain = 1./dampingtime
    #damper = IdealBunchFeedback(gain)


    # CREATE MONITORS
    # ===============

    try:
        bucket = machine.longitudinal_map.get_bucket(bunch)
    except AttributeError:
        bucket = machine.rfbucket

    simulation_parameters_dict = {
        'gamma'    : machine.gamma,
        'intensity': intensity,
        'Qx'       : machine.Q_x,
        'Qy'       : machine.Q_y,
        'Qs'       : bucket.Q_s,
        'beta_x'   : bunch.beta_Twiss_x(),
        'beta_y'   : bunch.beta_Twiss_y(),
        'beta_z'   : bucket.beta_z,
        'epsn_x'   : bunch.epsn_x(),
        'epsn_y'   : bunch.epsn_y(),
        'sigma_z'  : bunch.sigma_z(),
    }
    bunchmonitor = BunchMonitor(
        outputpath+'/bunchmonitor_{:04d}_chroma={:g}'.format(it, chroma),
        n_turns, simulation_parameters_dict,
        write_buffer_to_file_every=512,
        buffer_size=4096, mpi=True, filling_scheme=filling_scheme)
#    slicemonitor = SliceMonitor(
#        outputpath+'/slicemonitor_{:04d}_chroma={:g}_bunch_{:04d}'.format(it, chroma, bunch.bucket_id[0]),
#        n_turns_slicemon,
#        slicer_for_slicemonitor, simulation_parameters_dict,
#        write_buffer_to_file_every=1, buffer_size=n_turns_slicemon)


    # TRACKING LOOP
    # =============
    # machine.one_turn_map.append(damper)
    machine.one_turn_map.append(wake_field)


    # for slice statistics monitoring:
    s_cnt = 0
    monitorswitch = False

    print ('\n--> Begin tracking...\n')

    # GO!!!
    for i in range(n_turns):

        t0 = time.clock()

        # track the beam around the machine for one turn:
        machine.track(beam)

        bunch_list = beam.split_to_views()
        bunch = bunch_list[0]

        ex, ey, ez = bunch.epsn_x(), bunch.epsn_y(), bunch.epsn_z()
        mx, my, mz = bunch.mean_x(), bunch.mean_y(), bunch.mean_z()

        # monitor the bunch statistics (once per turn):
        bunchmonitor.dump(beam)

        # if the centroid becomes unstable (>1cm motion)
        # then monitor the slice statistics:
        if not monitorswitch:
            if mx > 1e-2 or my > 1e-2 or i > n_turns - n_turns_slicemon:
                print ("--> Activate slice monitor")
                monitorswitch = True
        else:
            if s_cnt < n_turns_slicemon:
#                slicemonitor.dump(bunch)
                s_cnt += 1

        # stop the tracking as soon as we have not-a-number values:
        if not all(np.isfinite(c) for c in [ex, ey, ez, mx, my, mz]):
            print ('*** STOPPING SIMULATION: non-finite bunch stats!')
            break

        # print status all 1000 turns:
        if i % 100 == 0:
            t1 = time.clock()
            print ('Emittances: ({:.3g}, {:.3g}, {:.3g}) '
                   '& Centroids: ({:.3g}, {:.3g}, {:.3g})'
                   '@ turn {:d}, {:g} ms, {:s}'.format(
                        ex, ey, ez, mx, my, mz, i, (t1-t0)*1e3, time.strftime(
                            "%d/%m/%Y %H:%M:%S", time.localtime()))
            )

    print ('\n*** Successfully completed!')


if __name__ == '__main__':
    # iteration, attached to monitor name:
    it = 0
    # outputpath relative to this file:
    outputpath = './'

    # run the simulation:
    chroma =0.
    intensity=1.1e11
    run(intensity=intensity, chroma=chroma, i_oct=0)
    
    import matplotlib.pyplot as plt
    

    if my_rank() == 0:
    
        h5f = h5py.File(outputpath+'/bunchmonitor_{:04d}_chroma={:g}.h5'.format(it, chroma),'r')
        
        data_mean_x = None
        data_mean_y = None
        data_mean_z = None
            
        
        for i, bunch_id in enumerate(filling_scheme):
            t_mean_x = h5f['Bunches'][str(bunch_id)]['mean_x'][:]
            t_epsn_x = h5f['Bunches'][str(bunch_id)]['epsn_x'][:]
            t_mean_y = h5f['Bunches'][str(bunch_id)]['mean_y'][:]
            t_mean_z = h5f['Bunches'][str(bunch_id)]['mean_z'][:]
        
            if data_mean_x is None:
                valid_map = (t_epsn_x > 0)
        
                turns = np.linspace(1,np.sum(valid_map),np.sum(valid_map))

        
                data_mean_x = np.zeros((np.sum(valid_map),len(filling_scheme)))
                data_mean_y = np.zeros((np.sum(valid_map),len(filling_scheme)))
                data_mean_z = np.zeros((np.sum(valid_map),len(filling_scheme)))
        
        
            np.copyto(data_mean_x[:,i],t_mean_x[valid_map])
            np.copyto(data_mean_y[:,i],t_mean_y[valid_map])
            np.copyto(data_mean_z[:,i],t_mean_z[valid_map]+-bunch_id)
        
        os.remove(outputpath+'/bunchmonitor_{:04d}_chroma={:g}.h5'.format(it, chroma))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        plot_n_turns = 50
        
        ax1.set_color_cycle([plt.cm.viridis(i) for i in np.linspace(0, 1, plot_n_turns)])
        ax2.set_color_cycle([plt.cm.viridis(i) for i in np.linspace(0, 1, plot_n_turns)])
        
        for i in range(plot_n_turns):
        
            ax1.plot(filling_scheme, data_mean_x[-(i+1),:]*1e3, '.')
            ax2.plot(filling_scheme, data_mean_y[-(i+1),:]*1e3, '.')
        
        
        ax1.set_xlabel('Bucket #')
        ax2.set_xlabel('Bucket #')
        
        ax1.set_ylabel('Bunch mean_x [mm]')
        ax2.set_ylabel('Bunch mean_y [mm]')
        plt.tight_layout()
        plt.show()