from nose.tools import assert_sequence_equal

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_less

# from .nmf import NMF
from sklearn.decomposition.nmf import NMF


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

    U = np.random.rand(n_samples, n_components)
    V = np.random.rand(n_components, n_features)

    # Normalise V
    Vnorm = np.linalg.norm(V, axis=1)
    V /= Vnorm[:, None]

    # Normalise U then order
    Unorm = np.linalg.norm(U, axis=0)
    U *= np.arange(n_components, 0, -1) / Unorm

    X = np.dot(U, V)
    assert_array_less(0, X)  # X is strictly greater than 0

    p = NMF(n_components, tol=1e-5)
    Ucalc = p.fit_transform(X)
    Vcalc = p.components_
    Xcalc = np.dot(Ucalc, Vcalc)

    # Check properties of Vcalc and Ucalc
    assert_sequence_equal(Ucalc.shape, (n_samples, n_components))
    assert_sequence_equal(Vcalc.shape, (n_components, n_features))
    assert_array_less(MAX_NEGATIVE_FLOAT, Ucalc)  # Ucalc >= 0
    assert_array_less(MAX_NEGATIVE_FLOAT, Vcalc)  # Vcalc >= 0

    assert_array_almost_equal(X, Xcalc, decimal=4)
#    assert_array_almost_equal(V, Vcalc)
#    assert_array_almost_equal(U, Ucalc)
