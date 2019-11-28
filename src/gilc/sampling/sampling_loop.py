from ..core.loop import Loop
from ..core.loop_state import LoopState

class SamplingLoop(Loop):
    def __init__(self):
        """ Base class for a core used for drawing samples """
        pass

    def compute_next_point(self, x0):
        """
        Computes the next sample from the linearly constrained unit Gaussian
        :param x0: current state
        :return: new state
        """
        return NotImplementedError

class SamplerState(LoopState):
    """
    Contains the state of the sampling loop, which includes a history of all evaluated locations
    """
    def __init__(self, x_init) -> None:
         self.samples = [x_init[:, i][:, None] for i in range(x_init.shape[-1])]
         self.iteration = 0

    def update(self, x_new) -> None:
        self.iteration += 1
        self.samples.append(x_new)

    @property
    def X(self):
        return np.hstack(self.samples)