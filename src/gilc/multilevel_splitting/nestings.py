import numpy as np

from .. import ShiftedLinearConstraints
from ..sampling import EllipticalSliceSampler

class Nesting():
    def __init__(self):
        """
        Base class for an individual nesting in a multilevel splitting method
        """
        pass

    def sample_from_nesting(self, n_samples, x_init, n_skip):
        """
        Draw samples from the nesting using LIN-ESS
        :param n_samples: number of samples to draw
        :param x_init: Starting point in domain
        :param n_skip: number of samples to skip in Markov chain
        :return: samples
        """
        return NotImplementedError

    def compute_log_nesting_factor(self, X):
        return NotImplementedError


class HDRNesting(Nesting):
    def __init__(self, linear_constraints, shift):
        """
        One nested domain for HDR
        :param linear_constraints: instance of linear constraints
        :param shift: shift defining the nesting
        """
        self.shifted_lincon = ShiftedLinearConstraints(linear_constraints.A, linear_constraints.b, shift)
        self.shift = shift
        self.dim = self.shifted_lincon.N_dim
        self.log_conditional_probability = None

        super().__init__()

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
        return sampler.loop_state.X

    def compute_log_nesting_factor(self, X):
        self.log_conditional_probability = np.log(self.shifted_lincon.integration_domain(X).sum()) - np.log(X.shape[1])


class SubsetNesting(Nesting):
    def __init__(self, linear_constraints, fraction, n_save=1):
        """
        Constructs a nesting given linear constraints and a fraction of samples that should lie inside the new nesting.
        Takes the samples as input since the subset is constructed directly from samples. Note the difference to
        HDRNesting, which is pre-constructed given shift values.
        :param fraction: Fraction of samples that should lie in the new domain
        :param linear_constraints: instance of LinearConstraints
        :param n_save: number of samples to save from inside the domain
        """

        self.fraction = fraction
        self.lincon = linear_constraints
        self.n_save = n_save

        # Compute subset properties from samples
        self.log_conditional_probability = None
        self.shift = None
        self.x_in = None
        self.shifted_lincon = None

        super().__init__()

    def update_properties_from_samples(self, X):
        """
        Computes the shift from samples and one sample within the domain
        :param X: Samples with shape (D, N)
        :return: None
        """
        n_inside = np.int(X.shape[-1] * self.fraction)

        # Update log conditional probability
        self.compute_log_nesting_factor(X)

        shiftvals = -np.amin(self.lincon.evaluate(X), axis=0)

        # pre-compute shift and index set
        if (shiftvals < 0).sum() > n_inside:
            # consider failure domain directly,
            self.shift = 0.
            idx_inside = self._update_fix_shift(self.shift, shiftvals)
        else:
            self.shift, idx_inside = self._update_find_shift(shiftvals)

        self.x_in = X[:, np.random.choice(idx_inside, size=self.n_save)].reshape(-1, 1)
        self.shifted_lincon = ShiftedLinearConstraints(self.lincon.A, self.lincon.b, self.shift)
        return

    def compute_log_nesting_factor(self, X):
        self.log_conditional_probability = np.log(np.int(X.shape[1] * self.fraction)) - np.log(X.shape[1])

    def sample_from_nesting(self, n, x_init, n_skip):
        """
        Draw samples from the nesting using LIN-ESS
        :param n: number of samples to draw
        :param x_init: Starting point in domain
        :param n_skip: number of samples to skip in Markov chain
        :return: samples
        """
        # sample from new domain using the elliptical slice sampler
        sampler = EllipticalSliceSampler(n, self.shifted_lincon, n_skip, x_init)
        sampler.run()
        n_init = x_init.shape[1]
        return sampler.loop_state.X[:, n_init:]

    def _update_find_shift(self, shiftvals):
        """
        Find the shift s.t. self.n_inside of the samples lie inside the new domain
        :param shiftvals: minimum of linear constraints evaluated at X
        :return: shift
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