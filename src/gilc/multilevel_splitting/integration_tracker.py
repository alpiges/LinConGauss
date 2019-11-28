import numpy as np

from ..core.loop_state import LoopState


class IntegratorState(LoopState):
    def __init__(self):
        self.nestings = []

    def add_nesting(self, new_nesting):
        """
        Add one nesting to the sequence of nestings
        :param new_nesting: New Nesting instance to add
        :return: None
        """
        self.nestings.append(new_nesting)

    def is_complete(self):
        """
        Checks if the nesting sequence is complete and covers the failure domain
        :return: Boolean
        """
        if len(self.sequence_of_nestings) == 0:
            return False
        return True if self.sequence_of_nestings[-1].shift == 0. else False

    def integral(self):
        return self.nesting_factors.prod()

    def log_integral(self):
        return self.log_nesting_factors.sum()

    def log2_integral(self):
        return self.log2_nesting_factors.sum()


class HDRTracker(IntegratorState):
    def __init__(self, shift_sequence):
        """
        Track record of HDR method given a sequence of shifts
        :param shift_sequence: sequence of shifts for HDR method (as np.ndarray)
        """
        super(SubsetSimulationTracker, self).__init__()
        self.shift_sequence = shift_sequence
        self.log_nesting_factors = self._get_log_nesting_factors()

    def add_nesting(self, hdr_nesting):
        """
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
        return self.log_nesting_factors / np.log(2.)

    def _get_log_nesting_factors(self):
        return np.asarray([nest.log_nesting_factor for nest in self.nestings])



class SubsetSimulationTracker(IntegratorState):
    def __init__(self):
        super(SubsetSimulationTracker, self).__init__()

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

    @property
    def shift_sequence(self):
        """
        All shifts in the nesting sequence
        :return: array of shifts
        """
        return np.asarray([nest.shift for nest in self.sequence_of_nestings])

    def x_inits(self):
        """ Initial locations of all nestings """
        return np.hstack([nest.x_in for nest in self.sequence_of_nestings])