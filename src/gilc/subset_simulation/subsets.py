import numpy as np
import time
from ..core import LinearConstraints
from ..loop import EllipticalSliceOuterLoop
from .nestings import SubsetSimRecords, NestedDomain

class SubsetSimulation():
    def __init__(self, linear_constraints, n_samples, domain_fraction, n_skip=0, timing=False):
        """
        Subset simulation to find a linearly constrained probability of failure in a Gaussian space
        :param linear_constraints: instance of LinearConstraints
        :param n_samples: number of samples per nesting (integer)
        :param domain_fraction: fraction of samples that should lie in the new domain (between 0 and 1)
        :param n_skip: number of samples to skip in ESS to get more independent samples
        :param timing: whether to measure and record loop runtime
        """
        self.lincon = linear_constraints
        self.n_samples = n_samples
        self.domain_fraction = domain_fraction
        self.dim = self.lincon.N_dim
        self.n_skip = n_skip

        # keep track of subset simulation
        self.tracker = SubsetSimRecords()

        # timing of every iteration in the loop
        self.timing = timing
        if self.timing:
            self.times = []

    def run_loop(self, verbose=True):
        """
        Run the subset sampling loop
        :param time: boolean whether to measure the time
        :param verbose: boolean whether to output current nesting number
        :return:
        """
        X = np.random.randn(self.dim, self.n_samples)
        subdomain = NestedDomain(X, self.domain_fraction, self.lincon)
        self.tracker.add_nesting(subdomain)

        count = 0
        while not self.tracker.is_complete():
            count += 1
            if self.timing:
                t = time.process_time()

            # sample from new domain using the elliptical slice sampler
            current_lincon = LinearConstraints(self.lincon.A, self.lincon.b + subdomain.shift)
            sampler = EllipticalSliceOuterLoop(self.n_samples, current_lincon, self.n_skip, subdomain.x_in)
            sampler.run_loop()

            # create new nesting and add it to records
            subdomain = NestedDomain(sampler.loop_state.X[:,1:], self.domain_fraction, self.lincon)
            self.tracker.add_nesting(subdomain)

            if self.timing:
                self.times.append(time.process_time()-t)
            if verbose:
                print('finished nesting #{}'.format(count))