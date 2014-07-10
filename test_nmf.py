from nose.tools import assert_sequence_equal

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_less

from .nmf import NMF


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

    p = NMF(n_components, tol=1e-5, max_iter=10000)
    Wcalc = p.fit_transform(X)
    Hcalc = p.components_
    Xcalc = np.dot(Wcalc, Hcalc)

    # Check properties of Hcalc and Wcalc
    assert_sequence_equal(Wcalc.shape, (n_samples, n_components))
    assert_sequence_equal(Hcalc.shape, (n_components, n_features))
    assert_array_less(MAX_NEGATIVE_FLOAT, Wcalc)  # Wcalc >= 0
    assert_array_less(MAX_NEGATIVE_FLOAT, Hcalc)  # Hcalc >= 0

    assert_array_almost_equal(X, Xcalc, decimal=4)
#    assert_array_almost_equal(H, Hcalc)
#    assert_array_almost_equal(W, Wcalc)
