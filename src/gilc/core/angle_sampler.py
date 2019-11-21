import numpy as np


class AngleSampler():
    def __init__(self, active_intersections):
        """
        Samples from a slice on an ellipse given through active intersections.
        :param active_intersections: ActiveIntersections object
        """
        self.active_intersections = active_intersections
        self.rotation_angle, self.rotated_slices = self.active_intersections.rotated_intersections()
        self.rotated_slices = self.rotated_slices.reshape(-1, 2)

    def draw_angle(self):
        """
        Draw one sample angle from the given slice
        :return: random angle from slice(s)
        """
        cum_len = self._get_slices_cumulative_length()
        l = cum_len[-1]

        sample = l*np.random.rand()   # random angle

        # which slice are we in?
        idx = np.searchsorted(cum_len, sample) - 1

        return self.rotated_slices[idx, 0] + sample - cum_len[idx] + self.rotation_angle

    def _get_slices_cumulative_length(self):
        """
        Compute the cumulative lengths of the slices, with a zero prepended
        :return: array with cumulative lengths of the slices
        """
        lengths = self.rotated_slices[:, 1] - self.rotated_slices[:, 0]
        cum_len = lengths.cumsum()
        return np.insert(cum_len, 0, 0)
