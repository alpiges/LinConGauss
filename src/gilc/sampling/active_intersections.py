import numpy as np


class ActiveIntersections():
    def __init__(self, ellipse, linear_constraints):
        """
        compute the intersections between an ellipse and M linear constraints
        and find those intersections that are on the boundary of the integration domain.
        :param ellipse: Ellipse instance
        :param linear_constraints: LinearConstraints instance
        """
        self.ellipse = ellipse
        self.lincon = linear_constraints
        self.N_constraints = self.lincon.b.shape[0]
        self.ellipse_in_domain = True

    def intersection_angles(self):
        """ Compute all of the up to 2M intersections of the ellipse and the linear constraints """
        g1 = np.dot(self.lincon.A, self.ellipse.a1)
        g2 = np.dot(self.lincon.A, self.ellipse.a2)

        r = np.sqrt(g1**2 + g2**2)
        phi = 2*np.arctan(g2/(r+g1)).squeeze()

        # two solutions per linear constraint, shape of theta: (M, 2)
        arg = - (self.lincon.b / r).squeeze()
        theta = np.zeros((self.N_constraints, 2))

        # write NaNs if there is no intersection
        arg = np.where(np.absolute(arg) <= 1, arg, np.nan)
        theta[:, 0] = np.arccos(arg) + phi
        theta[:, 1] = - np.arccos(arg) + phi
        theta = theta[np.isfinite(theta)]

        return np.sort(theta + (theta < 0.)*2.*np.pi)   # in [0, 2*pi]

    def find_active_intersections(self):
        """
        Find angles of those intersections that are at the boundary of the integration domain
        by adding and subtracting a small angle and evaluating on the ellipse to see if we are on the boundary of the
        integration domain.
        :return: angles of active intersection in order of increasing angle theta such that activation happens in
        positive direction. If a slice crosses theta=0, the first angle is appended at the end of the array.
        Every row of the returned array defines a slice for elliptical slice sampling.
        """
        delta_theta = 1.e-10 * 2.*np.pi
        theta = self.intersection_angles()

        active_directions = self._index_active(theta, delta_theta)
        theta_active = theta[np.nonzero(active_directions)]

        while theta_active.size % 2 == 1:
            # Almost tangential ellipses, reduce delta_theta
            delta_theta = 1.e-1 * delta_theta
            active_directions = self._index_active(theta, delta_theta)
            theta_active = theta[np.nonzero(active_directions)]

        if not theta_active.size:
            theta_active = np.asarray([0, 2*np.pi])
            if not self.lincon.integration_domain(self.ellipse.x(2 * np.pi * np.random.rand())):
                # entire ellipse is outside of the domain
                self.ellipse_in_domain = False
        else:
            if active_directions[np.nonzero(active_directions)][0] == -1:
                theta_active = np.append(theta_active[1:], theta_active[0])

        return theta_active

    def rotated_intersections(self):
        """
        Rotates the interactions by the first angle in self.slices (rotation angle)
        and makes sure that all angles lie in [0, 2*pi]
        :return: rotation angle (np.float), shifted angles (np.ndarray)
        """
        slices = self.find_active_intersections()
        rotation_angle = slices[0]
        slices = slices - rotation_angle

        return rotation_angle, slices + (slices < 0)*2.*np.pi

    def _index_active(self, t, dt):
        """
        Compute indices of angles on the ellipse that are on the boundary of the integration domain
        :param t: angle theta, shape (2M,)
        :param dt: infinitesimal angle delta_theta (integer)
        :return: indices where result is non-zero
        """
        idx = np.zeros_like(t)
        idx[:] = self.lincon.integration_domain(self.ellipse.x(t + dt)) - \
                 self.lincon.integration_domain(self.ellipse.x(t - dt))

        return idx



