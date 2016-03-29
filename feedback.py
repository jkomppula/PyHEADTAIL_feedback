import numpy as np
from scipy.constants import c, e
import scipy.integrate as integrate
import scipy.special as special
import itertools
from collections import deque
from transfer_functions import matrixGenerator



class PickUp(object):
    # The simplest possible feedback which correct a mean xp value of the bunch.
    def __init__(self,gain,register_length,slicer,transfer_function):
        self.gain = gain    # fraction of offset is corrected each
        self.slicer = slicer
        self.x_register = deque()
        self.y_register = deque()
        self.register_length = register_length

        self.mode, self.n_slices, _, _=slicer.config

        self.transfer_function = transfer_function
        self.transfer_matrix = []

    def track(self,bunch):
        slice_set = bunch.get_slices(self.slicer, statistics=['mean_x', 'mean_y','mean_z'])

        if len(self.transfer_matrix) == 0:
            self.matrix_generator = matrixGeneratorGenerator(self.transfer_function,slice_set)

        if len(self.transfer_matrix) == 0 or self.mode != 'uniform_bin':
            self.tranfer_matrix = self.matrix_generator(slice_set.z_bins,slice_set.mean_z)

        raw_signal_x=np.array([offset for offset in slice_set.mean_x])
        raw_signal_y=np.array([offset for offset in slice_set.mean_y])

        real_signal_x=np.dot(self.tranfer_matrix,raw_signal_x)
        real_signal_y=np.dot(self.tranfer_matrix,raw_signal_y)

        self.x_register.append(real_signal_x)
        self.y_register.append(real_signal_y)

        if len(self.x_register) > self.register_length:
            self.x_register.popleft()

        if len(self.y_register) > self.register_length:
            self.y_register.popleft()


class Kicker(object):
    def __init__(self,gain,slicer):
        self.gain=gain
        self.slicer = slicer
        self.pickups = []
        self.pickup_turns = []
        self.transfer_functions = []
        self.matrix_generators = []

        self.mode, self.n_slices, _, _=slicer.config
        self.n_pickups = 0
        self.transfer_matrices = []

    def add_pickup(self,pickup,turns,transfer_function):
        self.pickups.append(pickup)
        self.pickup_turns.append(turns)
        self.transfer_functions.append(transfer_function)
        self.n_pickups += 1


    def track(self,bunch):
        slice_set = bunch.get_slices(self.slicer, statistics=['mean_xp', 'mean_yp','mean_z'])

        if len(self.matrix_generators) == 0:
            for transfer_function in self.transfer_functions:
                self.matrix_generators.append(matrixGeneratorGenerator(transfer_function,slice_set))

        if len(self.transfer_matrices) == 0 or self.mode != 'uniform_bin':
            self.transfer_matrices= []

            for matrix_generator in self.matrix_generators:
                self.transfer_matrices.append(matrix_generator(slice_set.z_bins,slice_set.mean_z))


        xp_correction = np.zeros(self.n_slices)
        yp_correction = np.zeros(self.n_slices)
        for pickup, turns, matrix in zip(self.pickups,self.pickup_turns,self.transfer_matrices):
            x_avg_signal = np.sum(pickup.x_register[:turns],axis=0)/turns/
            y_avg_signal = np.sum(pickup.y_register[:turns],axis=0)/turns
            xp_correction = np.sum(xp_correction, np.dot(matrix,x_avg_signal)/self.n_pickups)
            yp_correction = np.sum(yp_correction, np.dot(matrix,y_avg_signal)/self.n_pickups)


        p_idx = slice_set.particles_within_cuts
        s_idx = slice_set.slice_index_of_particle.take(p_idx)

        # change xp and yp values of each macroparticle. The change corresponds correction signal values for each slice
        # in register
        for p_id, s_id in itertools.izip(p_idx,s_idx):
            bunch.xp[p_id] -= xp_correction[s_id]
            bunch.yp[p_id] -= yp_correction[s_id]

    def print_matrix(self,index):
        print 'Baa'

