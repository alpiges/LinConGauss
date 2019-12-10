import numpy as np
from .nestings import HDRNesting

class HDR():
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

        self.lincon = linear_constraints
        self.shift_sequence = shift_sequence
        self.n_samples = n_samples
        self.dim = self.lincon.N_dim
        self.X_init = X_init
        self.n_skip = n_skip
        self.tracker = HDRRecords(self.shift_sequence)
        self.X_domain = None

        # timing of every iteration in the core
        self.timing = timing
        if self.timing:
            self.times = []

    def run(self, save_samples=False, verbose=False):
        """
        Run the HDR method
        :return:
        """
        X = np.random.randn(self.dim, self.n_samples)

        for i, shift in enumerate(self.shift_sequence):
            if self.timing:
                t = time.process_time()

            current_nesting = HDRNesting(self.lincon, shift)
            current_nesting.compute_log_nesting_factor(X)
            if save_samples:
                current_nesting.save_X(X)

            self.tracker.add_nesting(current_nesting)

            X = current_nesting.sample_from_nesting(self.n_samples, self.X_init[:, i, None], self.n_skip)

            if self.timing:
                self.times.append(time.process_time() - t)
            if verbose:
                print('finished nesting #{}'.format(i))

        # save the samples that lie in the domain of interest
        # TODO: This is a bit hacky, one shouldn't need to draw these last samples...
        self.X_domain = X
        return X