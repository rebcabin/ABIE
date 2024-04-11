import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass, field


def two_particle_derivatives(x: np.ndarray,
                             t_: float,
                             params: dict) -> np.ndarray:
    """Book calls this 'ode_two_body_first_order.'
    Ignore t.'"""
    G = params['G']
    masses = params['masses']
    dxdt = np.zeros(len(x))
    R1 = x[0:3]
    R2 = x[3:6]
    r3 = np.linalg.norm(R2 - R1) ** 3
    dxdt[0:3] = x[6:9]
    dxdt[3:6] = x[9:12]
    dxdt[6:9] = - G * masses[0] * (R1 - R2) / r3
    dxdt[9:12] = - G * masses[1] * (R2 - R1) / r3
    return dxdt


@dataclass
class TwoBodyProblem:
    R1 = np.array([1., 0., 0.])
    R2 = np.array([-1., 0., 0.])
    V1 = np.array([0., 0.5, 0.])
    V2 = np.array([0., -0.5, 0.])
    masses = np.array([1., 1.])
    G = 1.0
    t0 = 0.0
    t1 = 100.0
    dt = 0.0001

    # The following attributes
    # are monkey-patched in. Big arrays are held for
    # platting after the integration.
    #
    #     x           (instantaneous state)
    #     dxdt        (instantaneous derivative of state)
    #     E           (instantaneous energy)
    #
    #     times       (all sampled time points)
    #     states      (all states)
    #     energies    (all energies)

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

    def _propagate_state(self):
        self.R1 = self.x[0:3]
        self.R2 = self.x[3:6]
        self.V1 = self.x[6:9]
        self.V2 = self.x[9:12]
        r3 = np.linalg.norm(self.R2 - self.R1) ** 3
        self.dxdt[0:3] = self.x[6:9]
        self.dxdt[3:6] = self.x[9:12]
        self.dxdt[6:9] = \
            - self.G * self.masses[0] \
            * (self.R1 - self.R2) / r3
        self.dxdt[9:12] = \
            - self.G * self.masses[1] \
            * (self.R2 - self.R1) / r3
        self.E = self.instantaneous_energy()

    def _init_euler(self):
        # instantaneous quantities
        self.x = np.concatenate((self.R1, self.R2, self.V1, self.V2))
        self.E = self.instantaneous_energy()
        state_len = len(self.x)
        self.dxdt = np.zeros(state_len)
        # plottable arrays
        self.npts = int((self.t1 - self.t0) / self.dt) + 1
        self.times = np.linspace(self.t0, self.t1, self.npts)
        self.states = np.zeros((self.npts, state_len))
        self.states[0, :] = self.x
        self.energies = np.zeros(self.npts)
        self.energies[0] = self.E

    def _step_euler(self):
        self.x += self.dxdt * self.dt

    def integrate_euler(self):
        self._init_euler()
        for e, t in enumerate(self.times):
            self._propagate_state()
            self._step_euler()
            self.states[e, :] = self.x
            self.energies[e] = self.E

    def plot_trajectory(self) -> None:
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        ax.plot(self.states[:, 0 + 0],
                self.states[:, 1 + 0], "b-")
        ax.plot(self.states[:, 0 + 3],
                self.states[:, 1 + 3], "g-")
        plt.show()

    def plot_energies(self):
        plt.figure()
        initial_energy = self.energies[0]
        relative_energies = np.abs((self.energies - initial_energy) / initial_energy)
        plt.semilogy(self.times, relative_energies)
        plt.xlabel('Time [arb]')
        plt.ylabel('|(E(t)-E0)/E0|')
        plt.show()
