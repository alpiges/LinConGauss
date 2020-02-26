import numpy as np

from .. import LoopState


class IntegratorState(LoopState):
    def __init__(self):
        super().__init__()
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
        return NotImplementedError

    def integral(self):
        return self.conditional_probabilities.prod()

    def log_integral(self):
        return self.log_conditional_probabilities.sum()

    def log2_integral(self):
        return self.log2_conditional_probabilities.sum()

    @property
    def conditional_probabilities(self):
        return np.exp(self.log_conditional_probabilities)

    @property
    def log2_conditional_probabilities(self):
        return self.log_conditional_probabilities / np.log(2.)

    @property
    def log_conditional_probabilities(self):
        return np.asarray([nest.log_conditional_probability for nest in self.nestings])


class HDRTracker(IntegratorState):
    def __init__(self, shift_sequence):
        """
        Track record of HDR method given a sequence of shifts
        :param shift_sequence: sequence of shifts for HDR method (as np.ndarray)
        """
        super().__init__()
        self.shift_sequence = shift_sequence
        # samples from domain of interest
        self.X = None

    def is_complete(self):
        return len(self.shift_sequence) == len(self.nestings)

    def add_samples(self, X):
        """ Save the samples from the domain of interest """
        self.X = X
        return


class SubsetSimulationTracker(IntegratorState):
    def __init__(self):
        super(SubsetSimulationTracker, self).__init__()

    def is_complete(self):
        """
        Checks if the nesting sequence is complete and covers the failure domain
        :return: Boolean
        """
        if len(self.nestings) == 0:
            return False
        return True if self.nestings[-1].shift == 0. else False

    def n_nestings(self):
        """
        :return: Number of nestings
        """
        return len(self.nestings)

    @property
    def shift_sequence(self):
        """
        All shifts in the nesting sequence
        :return: array of shifts
        """
        return np.asarray([nest.shift for nest in self.nestings])

    def x_inits(self):
        """ Initial locations needed for sampling from the nestings """
        return np.hstack([nest.x_in for nest in self.nestings])