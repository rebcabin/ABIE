import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class TwoBodyProblem:
    """The following attributes are monkey-patched in. Big
    arrays are held for platting after the integration. After
    a state update by any of the *_stepper methods, x is x_n.
    After a call of _propagate_state, R1, R2, V1, V2, E, L, e
    are current. State derivatives are computed on-the-fly via
    method _state_derivatives.

        x               12-vector  (instantaneous state)
        E               float      (instantaneous energy)
        L               3-vector   (instantaneous angular-momentum vector)
        e               3-vector   (instantaneous eccentricity vector)

    big arrays:

        times           N-vector   (all sampled time points)
        states          Nx12 array (all states)

        optional (controlled by flags):

        energies        N-vector   (all energies)
        angmoms         Nx3 array  (all angular-momentum vectors)
        eccentricities  Nx3 array  (all eccentricity vectors)

    """

    # Class constants
    G: ClassVar[float] = 1.0
    masses: ClassVar[np.ndarray] = np.array([1.0, 1.0])
    nbodies: ClassVar[int] = 2
    nstates: ClassVar[int] = 12

    # Instance constants: change in the caller to tailor an instance.
    t0: float = 0.0
    t1: float = 100.0
    dt: float = 0.0001

    # Initial Conditions; update these variables in
    # _propagate_state. Also change them in the caller.
    R1 = np.array([ 1.0,  0.0, 0.0])
    R2 = np.array([-1.0,  0.0, 0.0])
    V1 = np.array([ 0.0,  0.5, 0.0])
    V2 = np.array([ 0.0, -0.5, 0.0])

    # For 2-step methods, require one back derivative
    # in addition to the current state, x.
    fnm1 = np.zeros(nstates)

    # Flags for optional plots, strongly affect running speed.
    expensive_energy_tracking = True
    expensive_angular_momentum_tracking = True
    expensive_eccentricity_tracking = True

    #   ___                          _          _
    #  / __|___ _ __  _ __ _  _ __ _| |_ ___ __| |
    # | (__/ _ \ '  \| '_ \ || / _` |  _/ -_) _` |
    #  \___\___/_|_|_| .__/\_,_\__,_|\__\___\__,_|
    #   ___          |_|_            _             __
    #  / __|___ _ _  __| |_ __ _ _ _| |_ ___  ___ / _|
    # | (__/ _ \ ' \(_-<  _/ _` | ' \  _(_-< / _ \  _|
    #  \___\___/_||_/__/\__\__,_|_||_\__/__/ \___/_|
    #  __  __     _   _
    # |  \/  |___| |_(_)___ _ _
    # | |\/| / _ \  _| / _ \ ' \
    # |_|  |_\___/\__|_\___/_||_|

    def circular_period(self):
        """Period of the orbit under the assumption that the
        orbiting body is circular. Not valid for even moderate
        eccentricities.
        """
        m1 = self.masses[0]
        m2 = self.masses[1]
        r = np.linalg.norm(self.R1 - self.R2)
        result = 2 * np.pi \
                 * np.sqrt(r ** 3 / (self.G * (m1 + m2)))
        return result

    def instantaneous_energy(self) -> float:
        speed1 = np.linalg.norm(self.V1)
        speed2 = np.linalg.norm(self.V2)
        m1 = self.masses[0]
        m2 = self.masses[1]
        # kinetic energies
        K1 = 0.5 * m1 * speed1 ** 2
        K2 = 0.5 * m2 * speed2 ** 2
        # potential energy
        r = np.linalg.norm(self.R1 - self.R2)
        P = - self.G * m1 * m2 / r
        # total energy
        result = K1 + K2 + P
        return result

    def instantaneous_angular_momentum(self):
        m1 = self.masses[0]
        L1 = m1 * np.cross(self.R1, self.V1)
        m2 = self.masses[1]
        L2 = m2 * np.cross(self.R2, self.V2)
        result = L1 + L2
        return result

    def instantaneous_eccentricity_vector(self):
        G = self.G
        m1 = self.masses[0]
        m2 = self.masses[1]
        M = m1 + m2
        r1 = self.R1 - self.R2
        r = np.linalg.norm(r1)
        v1 = self.V1 - self.V2
        p1 = np.cross(v1, np.cross(r1, v1)) / (G * M)
        p2 = r1 / r
        result = p1 - p2
        return result

    #  ___                       _
    # |   \ _  _ _ _  __ _ _ __ (_)__ ___
    # | |) | || | ' \/ _` | '  \| / _(_-<
    # |___/ \_, |_||_\__,_|_|_|_|_\__/__/
    #       |__/

    def _state_derivatives(self, t, x):
        """Does not access stored state in instance variable
        x; takes state x as an input. In state-space form for
        two bodies, positions are in the first six positions
        of x and velocities are in the second six positions of
        x. Compute derivatives for any state, not necessarily
        at dt boundaries. Necessary for intermediate-point
        integrators like RK4.
        """
        R1 = x[0:3]
        R2 = x[3:6]
        V1 = x[6:9]
        V2 = x[9:12]
        f = np.zeros(len(x))
        f[0:3] = V1
        f[3:6] = V2
        r3 = np.linalg.norm(R2 - R1) ** 3
        f[6:9] = self._R1_double_dot(R1, R2, r3)
        f[9:12] = self._R2_double_dot(R1, R2, r3)
        return f

    def _R2_double_dot(self, R1, R2, r3):
        """Access only constants through self."""
        return - self.G * self.masses[1] \
            * (R2 - R1) / r3

    def _R1_double_dot(self, R1, R2, r3):
        """Access only constants through self."""
        return - self.G * self.masses[0] \
            * (R1 - R2) / r3

    # __   __       _     _     _        _
    # \ \ / /__ _ _| |___| |_  | |_  ___| |_ __
    #  \ V / -_) '_| / -_)  _| | ' \/ -_) | '_ \
    #   \_/\___|_| |_\___|\__| |_||_\___|_| .__/
    #                                     |_|

    def _verlet_F(self, t, y):
        """Do not access self variables x, times, or dt,
        rather work with input positions y. Access self only
        for methods *_double_dot.

        Compute second derivative of positions y with respect
        to time and return a six-vector. Works at any
        intermediate time-point, not only at dt boundaries.
        """
        R1 = y[0:3]
        R2 = y[3:6]
        F = np.zeros(len(y))
        r3 = np.linalg.norm(R2 - R1) ** 3
        F[0:3] = self._R1_double_dot(R1, R2, r3)
        F[3:6] = self._R2_double_dot(R1, R2, r3)
        return F

    def _velocity_verlet_update_positions(self, t, yn, yn_dot):
        """Equation 5.36a, page 75. Do not access stored state
        in instance variables. Take a temporary position
        6-vector and a temporary velocity 6-vector as input
        arguments. Access dt from self. Return a 6-vector of
        positions.
        """
        dt = self.dt
        ynp1 = np.copy(yn)
        ynp1 += yn_dot * dt
        ynp1 += self._verlet_F(t, yn) * (dt * dt / 2)
        return ynp1

    def _velocity_verlet_update_velocities(self, t, yn, yn_dot, ynp1):
        """Equation 5.36b, page 75. Do not access stored state
        in instance variables. Take a temporary state
        12-vector as an input argument. Access dt from self.
        Return a 6-vector of velocities. Call this after
        calling _vv_update_positions to get ynp1.
        """
        dt2 = self.dt/2
        ynp1_dot = np.copy(yn_dot)
        ynp1_dot += self._verlet_F(t, yn) * dt2
        ynp1_dot += self._verlet_F(t, ynp1) * dt2
        return ynp1_dot

    #  ___      _ _   _      _ _
    # |_ _|_ _ (_) |_(_)__ _| (_)______ _ _
    #  | || ' \| |  _| / _` | | |_ / -_) '_|
    # |___|_||_|_|\__|_\__,_|_|_/__\___|_|

    def _init_fixed_time_step(self):
        """Initialize data for any method of fixed-time step.
        The time step is stored in the pseudoconstant self.dt.
        """
        # initial conditions
        self.R1 = np.array([ 1., 0.0, 0.0])
        self.R2 = np.array([-1., 0.0, 0.0])
        self.V1 = np.array([ 0., 0.5, 0.0])
        self.V2 = np.array([0., -0.5, 0.0])

        # instantaneous quantities
        self.x = np.concatenate((self.R1, self.R2, self.V1, self.V2))
        self.E = self.instantaneous_energy()
        self.L = self.instantaneous_angular_momentum()
        self.e = self.instantaneous_eccentricity_vector()

        # plottable arrays
        self.npts = int((self.t1 - self.t0) / self.dt) + 1
        self.times = np.linspace(self.t0, self.t1, self.npts)
        state_len = len(self.x)
        self.states = np.zeros((self.npts, state_len))
        self.states[0, :] = self.x

        # optional arrays
        if self.expensive_energy_tracking:
            self.energies = np.zeros(self.npts)
            self.energies[0] = self.E

        if self.expensive_angular_momentum_tracking:
            angmom_len = 3
            self.angmoms = np.zeros((self.npts, angmom_len))
            self.angmoms[0, :] = self.L

        if self.expensive_eccentricity_tracking:
            eccentricity_vector_len = 3
            self.eccentricities = np.zeros((self.npts, eccentricity_vector_len))
            self.eccentricities[0] = self.e

    #  ___ _
    # / __| |_ ___ _ __ _ __  ___ _ _ ___
    # \__ \  _/ -_) '_ \ '_ \/ -_) '_(_-<
    # |___/\__\___| .__/ .__/\___|_| /__/
    #             |_|  |_|

    def euler_stepper(self, t):
        self.x += self._state_derivatives(t, self.x) * self.dt

    def rk4_stepper(self, t):
        dt = self.dt
        dt2 = dt / 2
        k1 = self._state_derivatives(t, self.x)
        k2 = self._state_derivatives(t + dt2, self.x + dt2 * k1)
        k3 = self._state_derivatives(t + dt2, self.x + dt2 * k2)
        k4 = self._state_derivatives(t + dt,  self.x + dt  * k3)
        self.x += dt * (k1 + (2 * k2) + (2 * k3) + k4) / 6

    def ab2_stepper(self, t):
        """2-step Adams-Bashforth algorithm"""
        if t == self.t0:
            self.fnm1 = self._state_derivatives(t, self.x)
            self.rk4_stepper(t)
        else:
            fn = self._state_derivatives(t, self.x)
            self.x += (self.dt / 2) * (3 * fn - self.fnm1)
            self.fnm1 = fn

    def verlet_stepper(self, t):
        yn = self.x[0:6]
        yn_dot = self.x[6:12]
        ynp1 = self._velocity_verlet_update_positions(t, yn, yn_dot)
        ynp1_dot = self._velocity_verlet_update_velocities(
            t, yn, yn_dot, ynp1)
        self.x[0:6] = ynp1
        self.x[6:12] = ynp1_dot

    #  ___     _                     _
    # |_ _|_ _| |_ ___ __ _ _ _ __ _| |_ ___ _ _
    #  | || ' \  _/ -_) _` | '_/ _` |  _/ _ \ '_|
    # |___|_||_\__\___\__, |_| \__,_|\__\___/_|
    #                 |___/

    def _propagate_state(self, t):
        """After a state update, propagate states to instance
        variables containing instantaneous values.
        """
        self.R1 = self.x[0:3]
        self.R2 = self.x[3:6]
        self.V1 = self.x[6:9]
        self.V2 = self.x[9:12]
        if self.expensive_energy_tracking:
            self.E = self.instantaneous_energy()
        if self.expensive_angular_momentum_tracking:
            self.L = self.instantaneous_angular_momentum()
        if self.expensive_eccentricity_tracking:
            self.e = self.instantaneous_eccentricity_vector()

    def integrate_fixed_time_step(self, stepper):
        self._init_fixed_time_step()
        for i, t in enumerate(self.times):
            stepper(t)
            self._propagate_state(t)
            self.states[i, :] = self.x
            if self.expensive_energy_tracking:
                self.energies[i] = self.E
            if self.expensive_angular_momentum_tracking:
                self.angmoms[i, :] = self.L
            if self.expensive_eccentricity_tracking:
                self.eccentricities[i, :] = self.e

    #  ___ _     _   _
    # | _ \ |___| |_| |_ ___ _ _ ___
    # |  _/ / _ \  _|  _/ -_) '_(_-<
    # |_| |_\___/\__|\__\___|_| /__/

    def plot_trajectory(self) -> None:
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        ax.plot(self.states[:, 0 + 0],
                self.states[:, 1 + 0], "b-")
        ax.plot(self.states[:, 0 + 3],
                self.states[:, 1 + 3], "g-")
        plt.show()

    def plot_energies(self):
        if self.expensive_energy_tracking:
            plt.figure()
            initial_energy = self.energies[0]
            relative_energies = np.abs(
                (self.energies - initial_energy) / initial_energy)
            plt.semilogy(self.times, relative_energies)
            plt.xlabel('Time [arb]')
            plt.ylabel('Energy: |(E(t)-E0)/E0|')
            plt.show()
        else:
            raise NotImplementedError('Energy was not tracked.')

    def plot_angular_momentum_magnitudes(self):
        if self.expensive_angular_momentum_tracking:
            plt.figure()
            L0 = self.angmoms[0]
            L0norm = np.linalg.norm(L0)
            Ls = [np.linalg.norm(L)
                  for L in np.abs((self.angmoms - L0) / L0norm)]
            plt.semilogy(self.times, Ls)
            plt.xlabel('Time [arb]')
            plt.ylabel('Angular Momentum: |(L(t)-L0)/L0|')
            plt.show()
        else:
            raise NotImplementedError('Angular momentum was not tracked.')

    def plot_eccentricity_magnitudes(self):
        if self.expensive_eccentricity_tracking:
            plt.figure()
            e0 = self.eccentricities[0]
            es = [np.linalg.norm(e)
                  for e in np.abs(self.eccentricities - e0)]
            plt.semilogy(self.times, es)
            plt.xlabel('Time [arb]')
            plt.ylabel('eccentricity: |(e(t) - e0) / e0|')
            plt.show()
        else:
            raise NotImplementedError('Eccentricity was not tracked.')
