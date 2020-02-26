import numpy as np

class LinearConstraints():
    def __init__(self, A, b, mode='Intersection'):
        """
        Defines linear functions f(x) = Ax + b.
        The integration domain is defined as the union of where all of these functions are positive if mode='Union'
        or the domain where any of the functions is positive, when mode='Intersection'
        :param A: matrix A with shape (M, D) where M is the number of constraints and D the dimension
        :param b: offset, shape (M, 1)
        """
        self.A = A
        self.b = b
        self.N_constraints = b.shape[0]
        self.N_dim = A.shape[1]
        self.mode = mode

    def evaluate(self, x):
        """
        Evaluate linear functions at N locations x
        :param x: location, shape (D, N)
        :return: Ax + b
        """
        return np.dot(self.A, x) + self.b

    def integration_domain(self, x):
        """
        is 1 if x is in the integration domain, else 0
        :param x: location, shape (D, N)
        :return: either self.indicator_union or self.indicator_intersection, depending on setting of self.mode
        """
        if self.mode == 'Union':
            return self.indicator_union(x)
        elif self.mode == 'Intersection':
            return self.indicator_intersection(x)
        else:
            raise NotImplementedError

    def indicator_intersection(self, x):
        """
        Intersection of indicator functions taken to be 1 when the linear function is >= 0
        :param x: location, shape (D, N)
        :return: 1 if all linear functions are >= 0, else 0.
        """
        return np.where(self.evaluate(x) >= 0, 1, 0).prod(axis=0)

    def indicator_union(self, x):
        """
        Union of indicator functions taken to be 1 when the linear function is >= 0
        :param x: location, shape (D, N)
        :return: 1 if any of the linear functions is >= 0, else 0.
        """
        return 1 - (np.where(self.evaluate(x) >= 0, 0, 1)).prod(axis=0)



class ShiftedLinearConstraints(LinearConstraints):
    def __init__(self, A, b, shift):
        """
        Class for shifted linear constraints that appear in multilevel splitting method
        :param A: matrix A with shape (M, D) where M is the number of constraints and D the dimension
        :param b: offset, shape (M, 1)
        :param shift: (positive) scalar value denoting the shift
        """
        self.shift = shift
        super(ShiftedLinearConstraints, self).__init__(A, b + shift)