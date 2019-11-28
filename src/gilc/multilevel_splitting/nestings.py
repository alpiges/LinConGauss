import numpy as np

from ..core import ShiftedLinearConstraints



class HDRNesting(NestedDomain):
    def __init__(self, linear_constraints, shift):
        """
        One nested domain for HDR
        :param linear_constraints: instance of linear constraints
        :param shift: shift defining the nesting
        """
        self.shifted_lincon = ShiftedLinearConstraints(linear_constraints.A, linear_constraints.b, shift)
        self.shift = shift
        self.dim = self.shifted_lincon.N_dim
        self.log_nesting_factor = None

    def samples_inside(self, X):
        """
        Boolean array whether samples X lie within the shifted domain
        :param X: samples
        :return: boolean array
        """
        return self.shifted_lincon.integration_domain(X)

    def n_inside(self, X):
        """
        Number of samples X that lie within the domain
        :param X: samples
        :return: number of samples inside domain
        """
        return self.samples_inside(X).sum()

    def sample_from_nesting(self, n_samples, x_init, n_skip):
        """
        Draw samples from the nesting using LIN-ESS
        :param n_samples: number of samples to draw
        :param x_init: Starting point in domain
        :param n_skip: number of samples to skip in Markov chain
        :return: samples
        """

        # sample from new domain using the elliptical slice sampler
        sampler = EllipticalSliceSampler(n_samples, self.shifted_lincon, n_skip, x_init)
        sampler.run()

        # create new nesting
        return sampler.loop_state.X

    def compute_log_nesting_factor(self, X):
        self.log_nesting_factor = np.log(self.n_inside(X)) - np.log(X.shape[1])



class SubsetNesting():
    def __init__(self, X, fraction, linear_constraints, keep_samples=True):
        """
        Constructs a nesting given linear constraints and a fraction of samples that should lie inside the new nesting.
        :param X: Samples with shape (D, N)
        :param fraction: Fraction of samples that should lie in the new domain
        :param linear_constraints: instance of LinearConstraints
        :param keep_samples: boolean, whether or not to save the samples (could be unfavorable due to memory)
        """
        self.keep_samples = keep_samples

        if self.keep_samples:
            self.X = X
        self.lincon = linear_constraints
        self.n_samples = X.shape[-1]
        self.n_inside = np.int(self.n_samples * fraction)

        shiftvals = -np.amin(self.lincon.evaluate(X), axis=0)

        # pre-compute shift and index set
        if (shiftvals<0).sum() > self.n_inside:
            # consider failure domain directly,
            self.shift = 0.
            self.idx_inside = self._update_fix_shift(self.shift, shiftvals)
        else:
            self.shift, self.idx_inside = self._update_find_shift(shiftvals)

        self.idx_minshift = self.idx_inside[np.argmin(shiftvals[self.idx_inside])]

        # Find the one sample that is furthest inside of the domain
        self.x_in = X[:, self.idx_minshift].reshape(-1, 1)

        # integral value
        self.conditional_probability = self.n_inside/self.n_samples

    def X_inside(self):
        """
        Find samples that are inside of the domain
        :return: samples inside the new domain or False if keep_samples==False
        """
        if self.keep_samples:
            return self.X[:, self.idx_inside]
        return False

    def compute_shift(self, X):
        """
        Find the shift value for given samples X
        :param X: Samples with shape (D, N)
        :return: shift (float)
        """

    def _update_find_shift(self, shiftvals):
        """
        Find the shift s.t. self.n_inside of the samples lie inside the new domain
        :param shiftvals: minimum of linear constraints evaluated at X
        :return: shift, indices of X that are inside, index of sample with highest shiftval
        """
        idx = np.argpartition(shiftvals, self.n_inside)[:self.n_inside + 1]
        shiftvals = shiftvals[idx]
        # shift = (shiftvals[-1] + np.amax(shiftvals[:-1])) / 2
        shift = shiftvals[-1]
        return shift, idx[:-1]

    def _update_fix_shift(self, shift, shiftvals):
        """
        Updates quantities once the shift becomes less than zero
        :param shift: value of shift
        :param shiftvals: minimum of linear constraints evaluated at X
        :return: indices with shiftvals larger than shift
        """
        idx = np.where(shiftvals<shift)[0]
        self.n_inside = (shiftvals<shift).sum()

        return idx