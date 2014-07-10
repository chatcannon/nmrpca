from nose.tools import assert_sequence_equal

import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_less,
                           assert_array_equal)

from . import nmf


def randn_complex(*shape):
    r = np.empty(shape, dtype=complex)
    r.real = np.random.randn(*shape)
    r.imag = np.random.randn(*shape)
    return r


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
