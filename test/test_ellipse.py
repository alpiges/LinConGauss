import numpy as np

from constrained_gaussian_integrals import Ellipse, LinearConstraints, ActiveIntersections, EllipticalSliceSampler
from constrained_gaussian_integrals.loop import EllipticalSliceOuterLoop


def test_ellipse_shape():
    D = 15 #dimension
    A = np.random.randn(D, 2)
    e = Ellipse(A[:, 0:1], A[:, 1:])

    theta = np.random.rand() * np.pi
    assert e.x(theta).shape == (D, 1)


def test_ellipse_samples_in_domain():
    """
    Tests whether samples of a fixed ellipse lie in integration domain

    Sets up a triangular domain (three constraints) around the center and an ellipse being a circle that intersects six
    times with the constraints.

    Then draw N samples from the ellipse and check if they all lie in the integration domain
    """
    A = np.asarray([[0, 1], [-np.sqrt(3), -1], [np.sqrt(3), -1]])
    b = np.sqrt(3) / 6. * np.asarray([[1., 2., 2]]).T

    lincon = LinearConstraints(A, b, mode='Intersection')
    ellipse = Ellipse(np.asarray([[1 / 3.], [0]]), np.asarray([[0], [1 / 3.]]))
    intersect = ActiveIntersections(ellipse, lincon)
    ess = EllipticalSliceSampler(intersect)

    N = 100
    samples = np.zeros((N,))
    for i in range(N):
        samples[i] = ess.draw_angle()

    x = ellipse.x(samples)
    assert lincon.integration_domain(x).prod() == 1.


def test_sampling():
    """
    Tests if all samples lie within the integral domain when using elliptical slice sampling
    """
    n_lc = 5
    n_dim = 2
    lincon = LinearConstraints(2 * np.random.randn(n_lc, n_dim), np.random.randn(n_lc, 1))
    sampler = EllipticalSliceOuterLoop(1000, lincon, n_skip=0)

    sampler.run_loop()
    assert np.all(lincon.integration_domain(sampler.loop_state.X)) == 1.

