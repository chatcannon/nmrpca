"""
Complex 'non-negative' matrix factorisation

Factorise complex matrices with bounds on the factors
"""
# Author: Chris Kerr <cjk34@cam.ac.uk>
#
# This file contains some code copied from scikit-learn's implementation in
# the file sklearn/decomposition/nmf.py, which has the following authors:
#         Vlad Niculae
#         Lars Buitinck <L.J.Buitinck@uva.nl>
#     original projected gradient NMF implementation:
#         Chih-Jen Lin, National Taiwan University
#     original Python and NumPy port:
#         Anthony Di Franco
#
# License: BSD 3 clause

import numpy as np
from numpy import linalg


class Constraint:
    """Constraint on the matrix factorisation (e.g. non-negativity for NMF)"""

    def project_H(self, H, copy=False):
        # do nothing
        return H

    def project_W(self, W, copy=False):
        # do nothing
        return W

    def normalise(self, W, H, copy=False):
        # do nothing
        return W, H


class NMFConstraint(Constraint):
    """Enforce the constraints in classic NMF"""

    def project_H(self, H, copy=False):
        if copy:
            H = np.copy(H)
        H[H < 0] = 0
        return H

    def project_W(self, W, copy=False):
        if copy:
            W = np.copy(W)
        W[W < 0] = 0
        return W


class NMFConstraint_NormaliseH(NMFConstraint):
    """Classic NMF except the H vectors are normalised"""

    def normalise(self, W, H, copy=False):
        if copy:
            W = np.copy(W)
            H = np.copy(H)
        Hnorm = linalg.norm(H, axis=1)
        H /= Hnorm[:, None]
        W *= Hnorm[None, :]
        return W, H


def svd_initialise(X, n_components, constraint):
    U, S, V = linalg.svd(X, full_matrices=False)
    W = U[:, :n_components] * np.sqrt(S[None, :n_components])
    H = V[:n_components, :] * np.sqrt(S[:n_components, None])

    W = constraint.project_W(W)
    H = constraint.project_H(H)
    W, H = constraint.normalise(W, H)
