import numpy as np

from LinConGauss import LinearConstraints
from LinConGauss.multilevel_splitting import SubsetSimulation, HDR

# define some linear constraints
n_lc = 5
n_dim = 3
np.random.seed(0) # because sometimes it is hard to find an initial point in the randomly drawn domain.
lincon = LinearConstraints(2 * np.random.randn(n_lc, n_dim), np.random.randn(n_lc, 1))

subset_simulator = SubsetSimulation(lincon, 16, 0.5)
subset_simulator.run(verbose=False)

shifts = subset_simulator.tracker.shift_sequence
x_inits = subset_simulator.tracker.x_inits()

hdr = HDR(lincon, shifts, 100, x_inits)
hdr.run()

def hdr_samples_in_domain():
    """ Test whether saved samples lie within the domain """
    assert np.all(lincon.integration_domain(hdr.tracker.X) == 1.)

def test_conditional_probability():
    """ Check that conditional probabilities lie between 0 and 1 """
    assert np.all(hdr.tracker.conditional_probabilities > 0.) and np.all(hdr.tracker.conditional_probabilities <= 1.)
