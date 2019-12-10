

class Loop():
    def __init__(self, n_iterations, linear_constraints, n_skip):
        """
        Base class for a loop
        :param n_iterations: Number of desired core iterations (integer)
        :param linear_constraints: an instance of LinearConstraints
        :param n_skip: number of samples to skip in order to get more independent samples
        """
        self.n_iterations = n_iterations
        self.lincon = linear_constraints
        self.n_skip = n_skip

    def run(self):
        """ Run the loop """
        return NotImplementedError
