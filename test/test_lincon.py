import numpy as np

from constrained_gaussian_integrals import LinearConstraints


def test_union_intersection():
    """ Tests whether union and intersection just have the opposite value of 0 and 1 """
    d = 15
    A = np.random.randn(d, d)
    b = np.zeros((d, 1))

    lincon = LinearConstraints(A, b)

    X = np.random.randn(d, 100)

    assert np.array_equal(lincon.indicator_intersection(X), 1-lincon.indicator_union(X))