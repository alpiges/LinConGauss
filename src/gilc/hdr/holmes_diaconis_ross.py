import numpy as np
from ..core import LinearConstraints
from ..loop import EllipticalSliceOuterLoop

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

        # timing of every iteration in the loop
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


class HDRRecords():
    def __init__(self, shift_sequence):
        """
        Track record of HDR method given a sequence of shifts
        :param shift_sequence: sequence of shifts for HDR method (as np.ndarray)
        """
        self.shift_sequence = shift_sequence
        self.nestings = []
        self.log_nesting_factors = self._get_log_nesting_factors()

    def add_nesting(self, hdr_nesting):
        """inte
        Add a nesting to the track record
        :param hdr_nesting: Instance of HDRNesting
        :return: None
        """
        self.nestings.append(hdr_nesting)
        self.log_nesting_factors = self._get_log_nesting_factors()
        return

    def is_complete(self):
        return len(self.shift_sequence) == len(self.nestings)

    def integral(self):
        return self.nesting_factors.prod()

    def log_integral(self):
        return self.log_nesting_factors.sum()

    def log2_integral(self):
        return self.log2_nesting_factors.sum()

    @property
    def nesting_factors(self):
        return np.exp(self.log_nesting_factors)

    @property
    def log2_nesting_factors(self):
        return self.log_nesting_factors/np.log(2.)

    def _get_log_nesting_factors(self):
        return np.asarray([nest.log_nesting_factor for nest in self.nestings])



class HDRNesting():
    def __init__(self, linear_constraints, shift):
        self.shifted_lincon = LinearConstraints(linear_constraints.A, linear_constraints.b + shift)
        self.shift = shift
        self.dim = self.shifted_lincon.N_dim
        self.X = None
        self.log_nesting_factor = None

    def compute_log_nesting_factor(self, X):
        self.log_nesting_factor = np.log(self.n_inside(X)) - np.log(X.shape[1])

    def idx_inside(self, X):
        """
        Indices of samples X that lie within the domain
        :param X: samples
        :return: index array
        """
        return self.shifted_lincon.integration_domain(X)

    def n_inside(self, X):
        """
        Number of samples X that lie within the domain
        :param X: samples
        :return: index array
        """
        return self.idx_inside(X).sum()

    def save_X(self, X):
        '''
        Keep the samples from the previous nesting (a fraction of which is in the current nesting)
        :param X: samples
        :return: None
        '''
        self.X = X
        return

    def sample_from_nesting(self, n_samples, x_init, n_skip):

        # sample from new domain using the elliptical slice sampler
        sampler = EllipticalSliceOuterLoop(n_samples, self.shifted_lincon, n_skip, x_init)
        sampler.run_loop()

        # create new nesting
        return sampler.loop_state.X

