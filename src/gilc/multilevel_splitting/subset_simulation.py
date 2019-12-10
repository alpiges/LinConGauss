import numpy as np
import time
from .nestings import SubsetNesting
from .integration_tracker import SubsetSimulationTracker

class SubsetSimulation():
    def __init__(self, linear_constraints, n_samples, domain_fraction, n_skip=0, timing=False):
        """
        Subset simulation to find a linearly constrained probability of failure in a Gaussian space
        :param linear_constraints: instance of LinearConstraints
        :param n_samples: number of samples per nesting (integer)
        :param domain_fraction: fraction of samples that should lie in the new domain (between 0 and 1)
        :param n_skip: number of samples to skip in ESS to get more independent samples
        :param timing: whether to measure and record core runtime
        """
        super().__init__(linear_constraints, n_samples, n_skip)

        self.domain_fraction = domain_fraction

        # keep track of subset simulation
        self.tracker = SubsetSimulationTracker()

        # timing of every iteration in the core
        self.timing = timing
        if self.timing:
            self.times = []

    def run(self, verbose=True):
        """
        Run the subset sampling core
        :param time: boolean whether to measure the time
        :param verbose: boolean whether to output current nesting number
        :return:
        """
        X = np.random.randn(self.dim, self.n_samples)
        subdomain = SubsetNesting(self.lincon, self.domain_fraction)
        subdomain.update_properties_from_samples(X)
        self.tracker.add_nesting(subdomain)

        count = 0
        while not self.tracker.is_complete():
            count += 1
            if self.timing:
                t = time.process_time()

            # sample from new domain using the elliptical slice sampler
            X = subdomain.sample_from_nesting(self.n_samples, subdomain.x_in, self.n_skip)

            # create new nesting and add it to records
            subdomain = SubsetNesting(self.lincon, self.domain_fraction)
            subdomain.update_properties_from_samples(X)
            self.tracker.add_nesting(subdomain)

            if self.timing:
                self.times.append(time.process_time()-t)
            if verbose:
                print('finished nesting #{}'.format(count))