import numpy as np


class SubsetSimRecords():
    def __init__(self):
        self.sequence_of_nestings = []

    def add_nesting(self, nested_domain):
        """
        Add one nesting to the sequence of nestings
        :param nested_domain:
        :return:
        """
        self.sequence_of_nestings.append(nested_domain)

    def integral(self):
        """
        Compute the integral of the smallest nesting
        :return: Integral value
        """
        integral = 1.
        for nest in self.sequence_of_nestings:
            integral *= nest.conditional_probability
        return integral

    def log_integral(self):
        """
        Compute the logarithm of the integral
        :return: Log integral value
        """
        log_integral = 0.
        for nest in self.sequence_of_nestings:
            log_integral += np.log(nest.conditional_probability)
        return log_integral

    def log2_integral(self):
        """
        Compute the logarithm to base 2 of the integral
        :return: Log2 integral value
        """
        log2_integral = 0.
        for nest in self.sequence_of_nestings:
            log2_integral += np.log2(nest.conditional_probability)
        return log2_integral

    def inner_domain_samples(self):
        """
        Returns samples from the smallest nesting
        :return: Samples of what is currently the inner nesting
        """
        return self.sequence_of_nestings[-1].X_inside()

    def is_complete(self):
        """
        Checks if the nesting sequence is complete and covers the failure domain
        :return: Boolean
        """
        if len(self.sequence_of_nestings) == 0:
            return False
        return True if self.sequence_of_nestings[-1].shift == 0. else False

    def n_nestings(self):
        """
        :return: Number of nestings
        """
        return len(self.sequence_of_nestings)

    def shifts(self):
        """
        All shifts in the nesting sequence
        :return: array of shifts
        """
        shifts = []
        for nest in self.sequence_of_nestings:
            shifts.append(nest.shift)
        return np.asarray(shifts)

    def x_inits(self):
        """ Initial locations of all nestings """
        x = []
        for nest in self.sequence_of_nestings:
            x.append(nest.x_in)
        return np.hstack(x)



class NestedDomain():
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