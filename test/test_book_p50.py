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
    two_body.t1 = 15.0
    two_body.expensive_angular_momentum_tracking = False

    with my_timeit('two_body_problem.solve') as euler_time:
        two_body.integrate_euler()

    print(f'\nIntegration time: {euler_time()} seconds')

    two_body.plot_trajectory()
    two_body.plot_energies()
    with pytest.raises(NotImplementedError) as excinfo:
        two_body.plot_angular_momentum_magnitudes()
    assert excinfo.value.args[0] == 'Angular momentum was not tracked.'


