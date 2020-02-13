import numpy as np

class Ellipse():
    def __init__(self, a1, a2):
        """
        Define an ellipse in a D-dimensional space around the origin
        from two vectors as x = a1 * cos(t) + a2 * sin(t)

        :param a1: first vector defining the ellipse, shape (D, 1)
        :param a2: second vector defining the ellipse, shape (D, 1)
        """
        self.a1 = a1
        self.a2 = a2

    def x(self, theta):
        """
        location on ellipse corresponding at angle theta
        :param theta: angle
        :return: location x on ellipse
        """
        return self.a1 * np.cos(theta) + self.a2 * np.sin(theta)

