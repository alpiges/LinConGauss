
import numpy as np

from .loop_state import LoopState
from ..core.ellipse import Ellipse
from ..core.elliptical_slice_sampling import EllipticalSliceSampler
from ..core.active_intersections import ActiveIntersections


class EllipticalSliceOuterLoop():
    def __init__(self, n_iterations, linear_constraints, n_skip, x_init=None):
        """
        Loop for sampling from a linearly constrained Gaussian
        :param n_iterations: Number of desired loop iterations (integer)
        :param linear_constraints: an instance of LinearConstraints
        :param n_skip: number of samples to skip in order to get more independent samples
        :param x_init: Initial sample(s) from domain of interest, np.ndarray with shape (dimension, number of samples)
        """
        self.n_iterations = n_iterations
        self.lincon = linear_constraints
        self.dim = self.lincon.N_dim
        self.n_skip = n_skip

        if x_init is None:
            # need to find a sample that lies in the domain :(
            found_sample = False
            while not found_sample:
                x_init = np.random.randn(self.dim, 1)
                if self.lincon.integration_domain(x_init):
                    found_sample = True

        self.loop_state = LoopState(x_init)

    def run_loop(self):
        """
        Sample from a linearly constrained unit Gaussian until stopping criterion is reached.
        :return: None
        """
        while not self.is_converged():
            x = self.loop_state.samples[-1]
            for i in range(self.n_skip + 1):
                x = self.compute_next_point(x)
                while not self.lincon.integration_domain(x):
                    print('Point outside domain, resample')
                    x = self.compute_next_point(self.loop_state.samples[-1])

            self.loop_state.update(x)

    def compute_next_point(self, x0):
        """
        Computes the next sample from the linearly constrained unit Gaussian
        :param x0: current state
        :return: new state
        """
        x1 = np.random.randn(self.lincon.N_dim, 1)
        ellipse = Ellipse(x0, x1)
        active_intersections = ActiveIntersections(ellipse, self.lincon)
        slice_sampler = EllipticalSliceSampler(active_intersections)

        if not active_intersections.ellipse_in_domain:
            # ellipse is outside of integration domain, reconstruct a new ellipse (should not happen at all!)
            raise ValueError('At least one point should be in the domain!')

        t_new = slice_sampler.draw_angle()
        return ellipse.x(t_new)

    def is_converged(self):
        """ Stopping criterion for sampling loop """
        return self.loop_state.iteration >= self.n_iterations
