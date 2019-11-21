import numpy as np


class LoopState():
    def __init__(self):
        """ Tracks important quantities during the loop """
        pass


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


class IntegratorState(LoopState):

class HDRTracker(IntegratorState):

class SubsetSimulationTracker(IntegratorState):