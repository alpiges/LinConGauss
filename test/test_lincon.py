import numpy as np

from LinConGauss import LinearConstraints, ShiftedLinearConstraints

# setting up linear constraints
d = 15
np.random.seed(0)
A = np.eye(d) + 0.5*np.random.randn(d,d)
b = np.random.rand(d,1)
lincon = LinearConstraints(A, b)

def test_lincon_evaluation():
    n = 100
    assert lincon.evaluate(np.random.randn(d, n)).shape == (d, n)

def test_union_intersection():
    """ Tests whether union and intersection just have the opposite value of 0 and 1 """
    X = np.random.randn(d, 100)
    assert np.array_equal(lincon.indicator_intersection(X), 1-lincon.indicator_union(X))

def test_shifted_lincon():
    shift = 1.
    shifted_lincon = ShiftedLinearConstraints(lincon.A, lincon.b, shift=shift)
    X = np.random.randn(d, 1)
    assert np.allclose(lincon.evaluate(X)+1., shifted_lincon.evaluate(X))