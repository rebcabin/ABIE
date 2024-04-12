import timeit
import pytest
from contextlib import contextmanager
import numpy as np

from examples.book_p50 import \
    (TwoBodyProblem)


@contextmanager
def my_timeit(name):
    start = stop = timeit.default_timer()
    yield lambda: stop - start
    stop = timeit.default_timer()


def test_two_body_problem():
    two_body = TwoBodyProblem()
    two_body.t1 = 100.0
    # two_body.expensive_angular_momentum_tracking = False

    with my_timeit('two_body_problem.solve') as integration_time:
        two_body.integrate_fixed_time_step(two_body.ab2_stepper)

    print(f'\nIntegration wall-clock time:   {integration_time()} seconds')
    print(f'Simulation time of last step:  {two_body.times[-1]} == {two_body.t1} seconds')
    print(f'orbital period in time units:  {two_body.circular_period()}')
    print(f'number of orbits:              {two_body.times[-1] / two_body.circular_period()}s')

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


