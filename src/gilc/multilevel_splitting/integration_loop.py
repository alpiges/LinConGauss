from ..core.loop import Loop

class IntegrationLoop(Loop):
    def __init__(self, linear_constraints, n_samples, domain_fraction, n_skip):
        """ Base class used for sampling based integration schemes (e.g. Subset Simulation, HDR)
        :param linear_constraints: instance of LinearConstraints
        :param n_samples: number of samples per nesting (integer)
        :param n_skip: number of samples to skip in ESS to get more independent samples
        """

        pass

    def run(self):
        pass
