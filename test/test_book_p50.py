import timeit
import pytest
from contextlib import contextmanager
import numpy as np
from dataclasses import dataclass

from examples.book_p50 import \
    (TwoBodyProblem)


@contextmanager
def my_timeit(name):
    start = stop = timeit.default_timer()
    yield lambda: stop - start
    stop = timeit.default_timer()


@dataclass
class Metrics:
    name: str = ""
    wall_clock_time: float = 0.0
    delta_time: float = 0.0
    max_energy_drift: float = 0.0
    max_ang_mom_drift: float = 0.0
    max_eccentricyt_drift: float = 0.0

    def take(self, two_body_problem: TwoBodyProblem):
        self.delta_time = two_body_problem.dt

        min_E = min(two_body_problem.energies)
        max_E = max(two_body_problem.energies)
        self.max_energy_drift = max_E - min_E

        angmom_magnitudes = \
            [np.linalg.norm(L)
             for L in two_body_problem.angmoms]
        min_L = min(angmom_magnitudes)
        max_L = max(angmom_magnitudes)
        self.max_ang_mom_drift = max_L - min_L

        eccentricity_magnitudes = \
            [np.linalg.norm(e)
             for e in two_body_problem.eccentricities]
        min_e = min(eccentricity_magnitudes)
        max_e = max(eccentricity_magnitudes)
        self.max_eccentricyt_drift = max_e - min_e

    def print(self):
        print(f'wall_clock_time:        {self.wall_clock_time:.3f}')
        print(f'delta_time:             {self.delta_time:.6f}')
        print(f'max_energy_drift:       {self.max_energy_drift:.3e}')
        print(f'max_ang_mom_drift:      {self.max_ang_mom_drift:.3e}')
        print(f'max_eccentricity_drift: {self.max_eccentricyt_drift:.3e}')


def test_compare_two_body_methods():
    two_body = TwoBodyProblem()
    two_body.t1 = 100.0
    two_body.dt = 0.001

    experiment(two_body, 'euler',  two_body.euler_stepper, False)
    experiment(two_body, 'rk4',  two_body.rk4_stepper, False)
    experiment(two_body, 'ab2',  two_body.ab2_stepper, False)
    experiment(two_body, 'verlet', two_body.verlet_stepper, False)


def experiment(two_body, name='', stepper=None, plot=True):
    print('\n%s' % name)
    with my_timeit('%s' % name) as time:
        two_body.integrate_fixed_time_step(stepper)
    metrics = Metrics(name=name, wall_clock_time=time())
    metrics.take(two_body)
    metrics.print()
    if plot:
        two_body.plot_trajectory()
        two_body.plot_energies()
        two_body.plot_angular_momentum_magnitudes()
        two_body.plot_eccentricity_magnitudes()
        pass


def test_two_body_problem():
    two_body = TwoBodyProblem()
    two_body.t1 = 100.0
    two_body.dt = 0.001
    # two_body.expensive_angular_momentum_tracking = False

    with my_timeit('two_body_problem.solve') as integration_time:
        two_body.integrate_fixed_time_step(
            two_body.euler_stepper)

    print(f'\nIntegration wall-clock time:   '
          f'{integration_time():.3f} seconds')

    print(f'time step:                     ' 
          f'{two_body.dt:.3f}')

    print(f'Simulation time of last step:  '
          f'{two_body.times[-1]:.3f} == {two_body.t1:.3f}')

    print(f'Orbital period in time units:  '
          f'{two_body.circular_period():.3f}')

    print(f'Number of orbits:              '
          f'{two_body.times[-1] / two_body.circular_period():.3f}')

    two_body.plot_trajectory()
    try:
        two_body.plot_energies()
    except NotImplementedError as e:
        print(e.args[0])

    try:
        two_body.plot_angular_momentum_magnitudes()
    except NotImplementedError as e:
        print(e.args[0])

    try:
        two_body.plot_eccentricity_magnitudes()
    except NotImplementedError as e:
        print(e.args[0])

    # with pytest.raises(NotImplementedError) as excinfo:
    #     two_body.plot_angular_momentum_magnitudes()
    # assert excinfo.value.args[0] == 'Angular momentum was not tracked.'
