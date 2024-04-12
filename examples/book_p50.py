import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass, field


@dataclass
class TwoBodyProblem:
    # Constants
    G = 1.0
    t0 = 0.0
    t1 = 100.0
    dt = 0.0001
    masses = np.array([1., 1.])
    nbodies = 2
    nstates = 12

    # Initial Conditions
    R1 = np.array([1., 0., 0.])
    R2 = np.array([-1., 0., 0.])
    V1 = np.array([0., 0.5, 0.])
    V2 = np.array([0., -0.5, 0.])

    # For 2-step methods, require one back derivative
    # in addition to the current state, x.
    fnm1 = np.zeros(nstates)

    # Flags affecting running speed.
    expensive_energy_tracking = True
    expensive_angular_momentum_tracking = True
    expensive_eccentricity_tracking = True

    # The following attributes
    # are monkey-patched in. Big arrays are held for
    # platting after the integration.
    #
    #     x               (instantaneous state)
    #     dxdt            (instantaneous derivative of state)
    #     E               (instantaneous energy)
    #     L               (instantaneous angular-momentum vector)
    #     e               (instantaneous eccentricity vector)
    #
    #     times           (all sampled time points)
    #     states          (all states)
    #     energies        (all energies)
    #     angmoms         (all angular-momentum vectors)
    #     eccentricities  (all eccentricity vectors)

    def circular_period(self):
        separation = np.linalg.norm(self.R1 - self.R2)
        m1 = self.masses[0]
        m2 = self.masses[1]
        result = 2 * np.pi * np.sqrt(separation ** 3 \
                                     / (self.G * (m1 + m2)))
        return result

    def instantaneous_angular_momentum(self):
        m1 = self.masses[0]
        m2 = self.masses[1]
        L1 = m1 * np.cross(self.R1, self.V1)
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

    def instantaneous_energy(self) -> float:
        speed1 = np.linalg.norm(self.V1)
        speed2 = np.linalg.norm(self.V2)
        separation = np.linalg.norm(self.R1 - self.R2)
        m1 = self.masses[0]
        m2 = self.masses[1]
        K1 = 0.5 * m1 * speed1 ** 2
        K2 = 0.5 * m2 * speed2 ** 2
        P = self.G * m1 * m2 / separation
        result = K1 + K2 - P
        return result

    def _propagate_state(self, t):
        self.R1 = self.x[0:3]
        self.R2 = self.x[3:6]
        self.V1 = self.x[6:9]
        self.V2 = self.x[9:12]
        if self.expensive_energy_tracking:
            self.E = self.instantaneous_energy()
        if self.expensive_eccentricity_tracking:
            self.e = self.instantaneous_eccentricity_vector()
        if self.expensive_angular_momentum_tracking:
            self.L = self.instantaneous_angular_momentum()

    def _state_derivatives(self, t, x):
        R1 = x[0:3]
        R2 = x[3:6]
        V1 = x[6:9]
        V2 = x[9:12]
        r3 = np.linalg.norm(R2 - R1) ** 3
        f = np.zeros(len(x))
        f[0:3] = V1
        f[3:6] = V2
        f[6:9] = \
            - self.G * self.masses[0] \
            * (R1 - R2) / r3
        f[9:12] = \
            - self.G * self.masses[1] \
            * (R2 - R1) / r3
        return f

    def _init_fixed_time_step(self):
        # instantaneous quantities
        self.x = np.concatenate((self.R1, self.R2, self.V1, self.V2))
        self.L = self.instantaneous_angular_momentum()
        self.e = self.instantaneous_eccentricity_vector()
        self.E = self.instantaneous_energy()
        state_len = len(self.x)

        # plottable arrays
        self.npts = int((self.t1 - self.t0) / self.dt) + 1
        self.times = np.linspace(self.t0, self.t1, self.npts)
        self.states = np.zeros((self.npts, state_len))
        self.states[0, :] = self.x

        if self.expensive_energy_tracking:
            self.energies = np.zeros(self.npts)
            self.energies[0] = self.E
        if self.expensive_eccentricity_tracking:
            eccentricity_vector_len = 3
            self.eccentricities = np.zeros((self.npts, eccentricity_vector_len))
            self.eccentricities[0] = self.e
        if self.expensive_angular_momentum_tracking:
            angmom_len = 3
            self.angmoms = np.zeros((self.npts, angmom_len))
            self.angmoms[0, :] = self.L

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

    def integrate_fixed_time_step(self, stepper):
        self._init_fixed_time_step()
        for e, t in enumerate(self.times):
            stepper(t)
            self._propagate_state(t)
            self.states[e, :] = self.x
            if self.expensive_energy_tracking:
                self.energies[e] = self.E
            if self.expensive_angular_momentum_tracking:
                self.eccentricities[e] = self.e
            if self.expensive_angular_momentum_tracking:
                self.angmoms[e, :] = self.L

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
            relative_energies = np.abs((self.energies - initial_energy) / initial_energy)
            plt.semilogy(self.times, relative_energies)
            plt.xlabel('Time [arb]')
            plt.ylabel('Energy: |(E(t)-E0)/E0|')
            plt.show()
        else:
            raise NotImplementedError('Energy was not tracked.')

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


