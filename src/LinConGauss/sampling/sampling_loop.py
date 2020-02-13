import numpy as np
from .. import Loop, LoopState


class SamplingLoop(Loop):
    def __init__(self, n_iterations, linear_constraints, n_skip):
        """
        Base class for a loop used for drawing samples
        """
        self.n_iterations = n_iterations
        super().__init__(linear_constraints, n_skip)

    def compute_next_point(self, x0):
        """
        Computes the next sample from the linearly constrained unit Gaussian
        :param x0: current state
        :return: new state
        """
        return NotImplementedError

    def is_converged(self):
        """ Stopping criterion for sampling core """
        return NotImplementedError


class SamplerState(LoopState):
    """
    Contains the state of the sampling loop, which includes a history of all evaluated locations
    """
    def __init__(self, x_init) -> None:
        self.samples = [x_init[:, i][:, None] for i in range(x_init.shape[-1])]
        self.iteration = 0
        super().__init__()

    def update(self, x_new) -> None:
        self.iteration += 1
        self.samples.append(x_new)

    @property
    def X(self):
        return np.hstack(self.samples)