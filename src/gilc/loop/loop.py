

class Loop():
    def __init__(self):
        """ Base class for a loop """
        pass

    def run(self):
        """ Run the loop """
        return NotImplementedError


class SamplingLoop(Loop):
    def __init__(self):
        """ Base class for a loop used for drawing samples """
        pass

    def compute_next_point(self, x0):
        """
        Computes the next sample from the linearly constrained unit Gaussian
        :param x0: current state
        :return: new state
        """
        return NotImplementedError


class IntegrationLoop(Loop):
    def __init__(self):
        """ Base class used for sampling based integration schemes (e.g. Subset Simulation, HDR) """
        pass

