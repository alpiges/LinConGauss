import numpy as np
import time

from .nestings import HDRNesting
from .integration_tracker import HDRTracker
from .integration_loop import IntegrationLoop


class HDR(IntegrationLoop):
    def __init__(self, linear_constraints, shift_sequence, n_samples, X_init, n_skip=0, timing=False):
        """
        Holmes-Diaconis-Ross algorithm for estimating integrals of linearly constrained Gaussians
        :param linear_constraints: instance of LinearConstraints
        :param shift_sequence: sequence of numbers > 0 that define the nestings (e.g. shifts from subset simulation)
        :param n_samples: number of samples per nesting (integer)
        :param X_init: starting points for ESS, the ith column has to be in the ith nesting
        :param n_skip: number of samples to skip in ESS
        :param timing: whether to measure the runtime
        """
        super().__init__(linear_constraints, n_samples, n_skip)

        self.shift_sequence = shift_sequence
        self.X_init = X_init
        self.tracker = HDRTracker(self.shift_sequence)

        # timing of every iteration in the core
        self.timing = timing
        if self.timing:
            self.times = []

    def run(self, verbose=False):
        """
        Run the HDR method
        :return:
        """
        for i, shift in enumerate(self.shift_sequence):
            if self.timing:
                t = time.process_time()

            if i == 0:
                X = np.random.randn(self.dim, self.n_samples)
            else:
                X = current_nesting.sample_from_nesting(self.n_samples, self.X_init[:, i, None], self.n_skip)

            current_nesting = HDRNesting(self.lincon, shift)
            current_nesting.compute_log_nesting_factor(X)
            self.tracker.add_nesting(current_nesting)

            if self.timing:
                self.times.append(time.process_time() - t)
            if verbose:
                print('finished nesting #{}'.format(i))

        # saving the samples from the domain of interest
        self.tracker.add_samples(X[:, self.lincon.integration_domain(X)==1])

    def draw_from_domain(self, n):
        """
        Sample from the domain of interest.
        :param n: number of samples to draw
        :return: samples (D, n)
        """
        domain = HDRNesting(self.lincon, 0.)
        return domain.sample_from_nesting(n, self.X_init[:, -1, None], self.n_skip)