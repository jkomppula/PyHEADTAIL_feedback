import numpy as np
from scipy.constants import m_p, c, e, pi
import matplotlib.pyplot as plt

from PyHEADTAIL.machines.synchrotron import Synchrotron



import PyHEADTAIL.particles.generators as generators
from PyHEADTAIL.trackers.transverse_tracking import TransverseMap
from PyHEADTAIL.trackers.simple_long_tracking import LinearMap
from PyHEADTAIL.particles.slicing import UniformBinSlicer


class Machine(Synchrotron):

    def __init__(self, n_segments = 1, Q_x=64.28, Q_y=59.31, Q_s=0.0020443):

        optics_mode='smooth'
        longitudinal_mode='non-linear'

        charge = e
        mass = m_p
        alpha = 53.86**-2
        h_RF = 35640

        p0 = 7000e9 * e / c
        p_increment = 0.
        accQ_x = Q_x
        accQ_y = Q_y
        V_RF = 16e6
        dphi_RF = 0

#        n_segments = kwargs['n_segments']
        circumference = 26658.883
        s = None
        alpha_x = None
        alpha_y = None
        self.beta_x = circumference / (2.*np.pi*accQ_x)
        self.beta_y = circumference / (2.*np.pi*accQ_y)
        D_x = 0
        D_y = 0

        # detunings
        Qp_x = 0
        Qp_y = 0

        app_x = 0
        app_y = 0
        app_xy = 0
        wrap_z=True

#        name = None
        self.sigma_z = 0.081
        self.epsn_x = 3.75e-6  # [m rad]
        self.epsn_y = 3.75e-6  # [m rad]
        self.intensity = 1.05e11

        super(Machine, self).__init__(
            optics_mode=optics_mode, circumference=circumference,
            n_segments=n_segments, s=s, name=None,
            alpha_x=alpha_x, beta_x=self.beta_x, D_x=D_x,
            alpha_y=alpha_y, beta_y=self.beta_y, D_y=D_y,
            accQ_x=accQ_x, accQ_y=accQ_y, Qp_x=Qp_x, Qp_y=Qp_y,
            app_x=app_x, app_y=app_y, app_xy=app_xy,
            alpha_mom_compaction=alpha, longitudinal_mode=longitudinal_mode,
            h_RF=np.atleast_1d(h_RF), V_RF=np.atleast_1d(V_RF),
            dphi_RF=np.atleast_1d(dphi_RF), p0=p0, p_increment=p_increment,
            charge=charge, mass=mass, wrap_z=wrap_z)


def generate_objects(machine,n_macroparticles, n_slices,n_sigma_z,
    filling_scheme = None, matched=True):
    print 'machine.sigma_z'
    print machine.sigma_z

    bunch = machine.generate_6D_Gaussian_bunch(
    n_macroparticles, machine.intensity, machine.epsn_x, machine.epsn_y, sigma_z=machine.sigma_z,
    filling_scheme=filling_scheme, matched=matched)

    slicer = UniformBinSlicer(n_slices=n_slices, n_sigma_z=n_sigma_z)

    return bunch, slicer, machine.transverse_map, machine.longitudinal_map


def track(n_turns, bunch, total_map, bunch_dump):
    for i in xrange(n_turns):
        bunch_dump.dump()

        for m_ in total_map:
            m_.track(bunch)


class BunchTracker(object):
    def __init__(self,bunch):
        self.counter = 0

        self.bunch = bunch
        self.turn = np.array([])

        self.mean_x = np.array([])
        self.mean_y = np.array([])
        self.mean_z = np.array([])

        self.mean_xp = np.array([])
        self.mean_yp = np.array([])
        self.mean_dp = np.array([])

        self.sigma_x = np.array([])
        self.sigma_y = np.array([])
        self.sigma_z = np.array([])

        self.sigma_xp = np.array([])
        self.sigma_yp = np.array([])
        self.sigma_dp = np.array([])

        self.epsn_x = np.array([])
        self.epsn_y = np.array([])
        self.epsn_z = np.array([])

    def dump(self):
        self.turn=np.append(self.turn,[self.counter])
        self.counter += 1

        self.mean_x=np.append(self.mean_x,[self.bunch.mean_x()])
        self.mean_y=np.append(self.mean_y,[self.bunch.mean_y()])
        self.mean_z=np.append(self.mean_z,[self.bunch.mean_z()])

        self.mean_xp=np.append(self.mean_xp,[self.bunch.mean_xp()])
        self.mean_yp=np.append(self.mean_yp,[self.bunch.mean_yp()])
        self.mean_dp=np.append(self.mean_dp,[self.bunch.mean_dp()])

        self.sigma_x=np.append(self.sigma_x,[self.bunch.sigma_x()])
        self.sigma_y=np.append(self.sigma_y,[self.bunch.sigma_y()])
        self.sigma_z=np.append(self.sigma_z,[self.bunch.sigma_z()])

        self.sigma_xp=np.append(self.sigma_xp,[self.bunch.sigma_xp()])
        self.sigma_yp=np.append(self.sigma_yp,[self.bunch.sigma_yp()])
        self.sigma_dp=np.append(self.sigma_dp,[self.bunch.sigma_dp()])

        self.epsn_x = np.append(self.epsn_x,[self.bunch.epsn_x()])
        self.epsn_y = np.append(self.epsn_y,[self.bunch.epsn_y()])
        self.epsn_z = np.append(self.epsn_z,[self.bunch.epsn_z()])


def compare_traces(trackers, labels):
    fig = plt.figure(figsize=(16, 8))
    ax_x_mean = fig.add_subplot(231)
    ax_x_sigma = fig.add_subplot(232)
    ax_x_epsn = fig.add_subplot(233)

    for i, tracker in enumerate(trackers):
        ax_x_mean.plot(tracker.turn, tracker.mean_x * 1000, label=labels[i])
    ax_x_mean.legend(loc='upper right')
    ax_x_mean.set_ylabel('<x> [mm]')
    ax_x_mean.ticklabel_format(useOffset=False)

    for i, tracker in enumerate(trackers):
        ax_x_sigma.plot(tracker.turn, tracker.sigma_x * 1000, label=labels[i])
    ax_x_sigma.set_ylabel('sigma_x [mm]')
    ax_x_sigma.ticklabel_format(useOffset=False)

    for i, tracker in enumerate(trackers):
        ax_x_epsn.plot(tracker.turn, tracker.epsn_x * 1e6, label=labels[i])
    ax_x_epsn.set_ylabel('epsn_x [mm mrad]')
    ax_x_epsn.set_xlabel('Turn')
    ax_x_epsn.ticklabel_format(useOffset=False)

    ax_y_mean = fig.add_subplot(234)
    ax_y_sigma = fig.add_subplot(235)
    ax_y_epsn = fig.add_subplot(236)

    for i, tracker in enumerate(trackers):
        ax_y_mean.plot(tracker.turn, tracker.mean_y * 1000, label=labels[i])
    ax_y_mean.legend(loc='upper right')
    ax_y_mean.set_ylabel('<y> [mm]')
    ax_y_mean.ticklabel_format(useOffset=False)

    for i, tracker in enumerate(trackers):
        ax_y_sigma.plot(tracker.turn, tracker.sigma_y * 1000, label=labels[i])
    ax_y_sigma.set_ylabel('sigma_y [mm]')
    ax_y_sigma.ticklabel_format(useOffset=False)

    for i, tracker in enumerate(trackers):
        ax_y_epsn.plot(tracker.turn, tracker.epsn_y * 1e6, label=labels[i])
    ax_y_epsn.set_ylabel('epsn_y  [mm mrad]')
    ax_y_epsn.set_xlabel('Turn')
    ax_y_epsn.ticklabel_format(useOffset=False)

    plt.show()


def compare_projections(bunches, labels):
    fig = plt.figure(figsize=(16, 4))
    fig.suptitle('z-x and z-y projections of bunches', fontsize=14, fontweight='bold')
    ax_z_x = fig.add_subplot(121)
    ax_z_y = fig.add_subplot(122)

    for i, bunch in enumerate(bunches):
        ax_z_x.plot(bunch.z, bunch.x * 1000, '.', label=labels[i])
    ax_z_x.legend(loc='upper right')
    ax_z_x.set_xlabel('z [m]')
    ax_z_x.set_ylabel('x [mm]')
    for i, bunch in enumerate(bunches):
        ax_z_y.plot(bunch.z, bunch.y * 1000, '.', label=labels[i])
    ax_z_y.legend(loc='upper right')
    ax_z_y.set_xlabel('z [m]')
    ax_z_y.set_ylabel('y [mm]')
