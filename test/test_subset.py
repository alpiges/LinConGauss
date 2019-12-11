import numpy as np

from gilc import LinearConstraints
from gilc.multilevel_splitting import SubsetSimulation

# define some linear constraints
n_lc = 5
n_dim = 3
np.random.seed(0) # because sometimes it is hard to find an initial point in the randomly drawn domain.
lincon = LinearConstraints(2 * np.random.randn(n_lc, n_dim), np.random.randn(n_lc, 1))

subset_simulator = SubsetSimulation(lincon, 16, 0.5)
subset_simulator.run(verbose=False)


def subset_finds_domain():
    """ Test whether subset simulation finds a sample in the domain of interest. """
    assert lincon.integration_domain(subset_simulator.tracker.x_inits()[:,-1]) == 1.


def shifts_larger_zero():
    """ Test that shifts found by subset simulation are greater/equal to 0 """
    pass


def sampling_from_subset():
    """ Test that samples from subset lie in the current domain """
    pass

