from nose.tools import assert_almost_equal, assert_sequence_equal

import numpy as np
from numpy.testing import assert_array_almost_equal

from . import pca


def assert_almost_diagonal(A, assert_real=False):
    Adiag = np.diag(A)
    if assert_real:
        Adiag = Adiag.real
    assert_array_almost_equal(A, np.diag(Adiag))


def randn_complex(*shape):
    r = np.empty(shape, dtype=complex)
    r.real = np.random.randn(*shape)
    r.imag = np.random.randn(*shape)
    return r


def test_PCA_real():
    n_samples = 10
    n_features = 20
    n_components = 3

    U = np.random.randn(n_samples, n_components)
    V = np.random.randn(n_components, n_features)

    # Make V orthonormal
    for i in range(n_components):
        for j in range(i):
            i_dot_j = np.dot(V[i, :], V[j, :])
            # N.B. V_j is already normalised
            V[i, :] -= i_dot_j * V[j, :]
        Vmean = np.mean(V[i, :])
        V[i, :] -= Vmean
        Vnorm = np.linalg.norm(V[i, :])
        V[i, :] /= Vnorm

    # Make U orthonormal, then multiply to get ordering of components
    for i in range(n_components):
        for j in range(i):
            i_dot_j = np.dot(U[:, i], U[:, j])
            # N.B. U_j is already normalised
            U[:, i] -= i_dot_j * U[:, j]
        Umean = np.mean(U[:, i])
        U[:, i] -= Umean
        Unorm = np.linalg.norm(U[:, i])
        U[:, i] /= Unorm
        # ensure ordering
        U[:, i] *= (n_components - i)

    X = np.dot(U, V)
    assert_almost_equal(0, np.mean(X))

    p = pca.PCA(n_components)
    Ucalc = p.fit_transform(X)
    Vcalc = p.components_
    Xcalc = np.dot(Ucalc, Vcalc)

    # Check properties of Vcalc and Ucalc
    assert_sequence_equal(Ucalc.shape, U.shape)
    assert_sequence_equal(Vcalc.shape, V.shape)
    assert_array_almost_equal(np.eye(n_components),
                              np.dot(Vcalc, Vcalc.T))
    assert_almost_diagonal(np.dot(Ucalc.T, Ucalc))
#    assert_array_almost_equal(np.diag(np.arange(n_components, 0, -1) ** 2),
#                              np.dot(Ucalc.T, Ucalc))

    assert_array_almost_equal(X, Xcalc)
#    assert_array_almost_equal(V, Vcalc)
#    assert_array_almost_equal(U, Ucalc)


def test_PCA_complex():
    n_samples = 10
    n_features = 20
    n_components = 3

    U = randn_complex(n_samples, n_components)
    V = randn_complex(n_components, n_features)

    # Make V orthonormal
    for i in range(n_components):
        for j in range(i):
            i_dot_j = np.dot(V[i, :], V[j, :])
            # N.B. V_j is already normalised
            V[i, :] -= i_dot_j * V[j, :]
        Vmean = np.mean(V[i, :])
        V[i, :] -= Vmean
        Vnorm = np.linalg.norm(V[i, :])
        V[i, :] /= Vnorm

    # Make U orthonormal, then multiply to get ordering of components
    for i in range(n_components):
        for j in range(i):
            i_dot_j = np.dot(U[:, i], U[:, j])
            # N.B. U_j is already normalised
            U[:, i] -= i_dot_j * U[:, j]
        Umean = np.mean(U[:, i])
        U[:, i] -= Umean
        Unorm = np.linalg.norm(U[:, i])
        U[:, i] /= Unorm
        # ensure ordering
        U[:, i] *= (n_components - i)

    X = np.dot(U, V)
    assert_almost_equal(0, np.mean(X))

    p = pca.PCA(n_components)
    Ucalc = p.fit_transform(X)
    Vcalc = p.components_
    Xcalc = np.dot(Ucalc, Vcalc)

    # Check properties of Vcalc and Ucalc
    assert_sequence_equal(Ucalc.shape, U.shape)
    assert_sequence_equal(Vcalc.shape, V.shape)
    assert_array_almost_equal(np.eye(n_components),
                              np.dot(Vcalc, np.conj(Vcalc.T)))
    assert_almost_diagonal(np.dot(np.conj(Ucalc.T), Ucalc), assert_real=True)
#    assert_array_almost_equal(np.diag(np.arange(n_components, 0, -1) ** 2),
#                              np.dot(Ucalc.T, Ucalc))

    assert_array_almost_equal(X, Xcalc)
#    assert_array_almost_equal(V, Vcalc)
#    assert_array_almost_equal(U, Ucalc)
