from nose.tools import assert_sequence_equal

import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_less,
                           assert_array_equal)

from .. import nmf


def randn_complex(*shape):
    r = np.empty(shape, dtype=complex)
    r.real = np.random.randn(*shape)
    r.imag = np.random.randn(*shape)
    return r

EPS = 1e-7
MAX_NEGATIVE_FLOAT = -5e-324


def test_NMF_real():
    n_samples = 10
    n_features = 20
    n_components = 3

    W = np.random.rand(n_samples, n_components)
    H = np.random.rand(n_components, n_features)

    # Normalise H
    Hnorm = np.linalg.norm(H, axis=1)
    H /= Hnorm[:, None]

    # Normalise U then order
    Wnorm = np.linalg.norm(W, axis=0)
    W *= np.arange(n_components, 0, -1) / Wnorm

    X = np.dot(W, H)
    assert_array_less(0, X)  # X is strictly greater than 0

    p = nmf.NMF(n_components, tol=1e-5, max_iter=1000)
    Wcalc = p.fit_transform(X)
    Hcalc = p.components_
    Xcalc = np.dot(Wcalc, Hcalc)

    # Check properties of Hcalc and Wcalc
    assert_sequence_equal(Wcalc.shape, (n_samples, n_components))
    assert_sequence_equal(Hcalc.shape, (n_components, n_features))
    assert_array_less(MAX_NEGATIVE_FLOAT, Wcalc)  # Wcalc >= 0
    assert_array_less(MAX_NEGATIVE_FLOAT, Hcalc)  # Hcalc >= 0

    assert_array_almost_equal(X, Xcalc, decimal=3)
#    assert_array_almost_equal(H, Hcalc)
#    assert_array_almost_equal(W, Wcalc)


def test_NMF_real_normalised():
    n_samples = 10
    n_features = 20
    n_components = 3

    W = np.random.rand(n_samples, n_components)
    H = np.random.rand(n_components, n_features)

    # Normalise H
    Hnorm = np.linalg.norm(H, axis=1)
    H /= Hnorm[:, None]

    # Normalise U then order
    Wnorm = np.linalg.norm(W, axis=0)
    W *= np.arange(n_components, 0, -1) / Wnorm

    X = np.dot(W, H)
    assert_array_less(0, X)  # X is strictly greater than 0

    p = nmf.ProjectedGradientNMF(n_components, nmf.NMFConstraint_NormaliseH(),
                                 max_iter=1000)
    Wcalc = p.fit_transform(X)
    Hcalc = p.components_
    Xcalc = np.dot(Wcalc, Hcalc)

    # Check properties of Hcalc and Wcalc
    assert_sequence_equal(Wcalc.shape, (n_samples, n_components))
    assert_sequence_equal(Hcalc.shape, (n_components, n_features))
    assert_array_less(MAX_NEGATIVE_FLOAT, Wcalc)  # Wcalc >= 0
    assert_array_less(MAX_NEGATIVE_FLOAT, Hcalc)  # Hcalc >= 0
    # Check that Hcalc is normalised
    assert_array_almost_equal(np.linalg.norm(Hcalc, axis=1), 1)
    # N.B. Hcalc is not orthogonal so can't assert H.H' == I

    assert_array_almost_equal(X, Xcalc, decimal=3)
#    assert_array_almost_equal(H, Hcalc)
#    assert_array_almost_equal(W, Wcalc)


def test_NMF_complex():
    n_samples = 10
    n_features = 20
    n_components = 3

    W = randn_complex(n_samples, n_components)
    H = np.random.rand(n_components, n_features) + 0j

    # Normalise H
    Hnorm = np.linalg.norm(H, axis=1)
    H /= Hnorm[:, None]

    # Normalise W then order
    Wnorm = np.linalg.norm(W, axis=0)
    W *= np.arange(n_components, 0, -1) / Wnorm

    X = np.dot(W, H)

    p = nmf.PinvNMF(n_components, nmf.ComplexMFConstraint(), max_iter=1000)
    Wcalc = p.fit_transform(X)
    Hcalc = p.components_
    Xcalc = np.dot(Wcalc, Hcalc)

    # Check properties of Hcalc and Wcalc
    assert_sequence_equal(Wcalc.shape, (n_samples, n_components))
    assert_sequence_equal(Hcalc.shape, (n_components, n_features))
    assert_array_less(MAX_NEGATIVE_FLOAT, Hcalc.real)  # Hcalc >= 0
    assert_array_equal(0, Hcalc.imag)  # Hcalc >= 0
    # Check that Hcalc is normalised
    assert_array_almost_equal(np.linalg.norm(Hcalc, axis=1), 1)
    # N.B. Hcalc is not orthogonal so can't assert H.H' == I

    assert_array_almost_equal(X, Xcalc, decimal=3)
#    assert_array_almost_equal(H, Hcalc)
#    assert_array_almost_equal(W, Wcalc)


def test_FIDConstraint_H():
    n_features = 20
    n_components = 3

    fidc = nmf.FIDConstraint()
    H = randn_complex(n_components, n_features)
    Horig = np.copy(H)

    Hc = fidc.project_H(H, copy=True)
    assert_array_equal(H, Horig)
    Hft = np.dual.fft(Hc, axis=1)
    assert_array_less(-EPS, Hft.real)
    # TODO check full KK relation; just check it sums to 0 for now
    assert_array_almost_equal(0, np.sum(Hft.imag, axis=1))

    # Check that the projection is idempotent
    Hcc = fidc.project_H(Hc, copy=True)
    assert_array_almost_equal(Hc, Hcc)

    Hc2 = fidc.project_H(H)
    assert_array_almost_equal(Hc, Hc2)
    assert Hc2 is H


def test_PhaseRange_W():
    n_samples = 200
    n_components = 50

    fidc = nmf.PhaseRangeFIDConstraint(np.pi / 4)
    W = randn_complex(n_samples, n_components)
    W += (1j - np.mean(W))  # make the average value to be 1j
    Worig = np.copy(W)

    Wc = fidc.project_W(W, copy=True)
    assert_array_equal(W, Worig)
    # phase must be (almost) between pi/4 and 3pi/4
    assert_array_less(0.24 * np.pi, np.angle(Wc))
    assert_array_less(np.angle(Wc), 0.76 * np.pi)

    # Check that the projection is idempotent
    Wcc = fidc.project_W(Wc, copy=True)
    # Projection changes the mean, and the mean is used as the centre of the
    # range, so projection is only approximately idempotent
    assert_array_almost_equal(Wc, Wcc, decimal=2)


def test_SamplePhase_W():
    n_samples = 200
    n_components = 50

    fidc = nmf.SamplePhaseFIDConstraint()
    W = randn_complex(n_samples, n_components)
    W += (1j - np.mean(W))  # make the average value to be 1j
    Worig = np.copy(W)

    Wc = fidc.project_W(W, copy=True)
    assert_array_equal(W, Worig)
    # phase must be (almost) the same for all components of a sample
    # however some elements are zero so we can't use the simple option
    Wsum = np.sum(Wc, axis=1)
    assert_array_almost_equal(np.abs(Wc + Wsum[:, None]),
                              np.abs(Wc) + np.abs(Wsum[:, None]))

    # Check that the projection is idempotent
    Wcc = fidc.project_W(Wc, copy=True)
    assert Wcc is not Wc  # Check that they aren't the same array
    assert_array_almost_equal(Wc, Wcc)
