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

import warnings

import numpy as np
from numpy import linalg
from numpy.dual import fft, ifft


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
        H[H <= 0] = 0  # Use <= 0 to set negative 0 to positive 0
        return H

    def project_W(self, W, copy=False):
        if copy:
            W = np.copy(W)
        W[W <= 0] = 0  # Use <= 0 to set negative 0 to positive 0
        return W


class NMFConstraint_NormaliseH(NMFConstraint):
    """Classic NMF except the H vectors are normalised"""

    def __init__(self, max_normalisation_factor=1e3):
        self.max_norm_fac = max_normalisation_factor

    def normalise(self, W, H, copy=False):
        if copy:
            W = np.copy(W)
            H = np.copy(H)
        Hnorm = linalg.norm(H, axis=1)
        Hnorm[Hnorm > self.max_norm_fac] = self.max_norm_fac
        Hnorm[Hnorm < 1/self.max_norm_fac] = 1/self.max_norm_fac
        H /= Hnorm[:, None]
        W *= Hnorm[None, :]
        return W, H


class ComplexMFConstraint(Constraint):
    """H is still constrained to be positive real, W can take any value"""

    def project_H(self, H, copy=False):
        if copy:
            H = np.copy(H)
        H.imag = 0  # Keep only the real part
        H[H.real <= 0] = 0  # Use <= 0 to set negative 0 to positive 0
        return H

    # No constraint on W

    def normalise(self, W, H, copy=False):
        if copy:
            W = np.copy(W)
            H = np.copy(H)
        Hnorm = linalg.norm(H, axis=1)
#        Hnorm[Hnorm > self.max_norm_fac] = self.max_norm_fac
#        Hnorm[Hnorm < 1/self.max_norm_fac] = 1/self.max_norm_fac
        H /= Hnorm[:, None]
        W *= Hnorm[None, :]
        return W, H


class FIDConstraint(Constraint):
    """Constraint for matrix factorisation of NMR Free Induction Decays

    The real part of the frequency space spectrum should be non-negative, and
    the imaginary part should be related to the real part by the Kramers-
    Kronig formula."""

    def project_H(self, H, copy=False):
        Hft = fft(H, axis=1)
        Hft.imag = 0  # Keep only the real part
        Hft[Hft.real <= 0] = 0
        if copy:
            H = ifft(Hft, axis=1)
        else:
            H[:, :] = ifft(Hft, axis=1)
        N = H.shape[1]
        KKfactor = np.linspace(2, 0, N, endpoint=False)
        KKfactor[0] = 1
        H *= KKfactor[None, :]
        return H

    def normalise(self, W, H, copy=False):
        if copy:
            W = np.copy(W)
            H = np.copy(H)
        Hnorm = linalg.norm(H, axis=1)
#        Hnorm[Hnorm > self.max_norm_fac] = self.max_norm_fac
#        Hnorm[Hnorm < 1/self.max_norm_fac] = 1/self.max_norm_fac
        H /= Hnorm[:, None]
        W *= Hnorm[None, :]
        return W, H


class PhaseRangeFIDConstraint(FIDConstraint):
    def __init__(self, angle_limit=np.pi/4):
        self.angle_limit = angle_limit

    def project_W(self, W, copy=False):
        """The phase of each component should be similar"""
        if copy:
            W = np.copy(W)
        Wsum = np.sum(W)
        Wfrac = W / Wsum
        Warg = np.angle(Wfrac)
        Wfabs = np.abs(Wfrac)

        arg_too_high = (Warg > self.angle_limit)
        arg_too_low = (Warg < -self.angle_limit)
        Wleft = Wsum * np.exp(1j * self.angle_limit)
        Wright = Wsum * np.exp(-1j * self.angle_limit)
        W[arg_too_high] = Wleft * Wfabs[arg_too_high]
        W[arg_too_low] = Wright * Wfabs[arg_too_low]

        return W


class SamplePhaseFIDConstraint(FIDConstraint):
    """For a given sample, all components should have the same phase"""

    def project_W(self, W, copy=False):
        # No need to copy
        Wsum = np.sum(W, axis=1)
        Wfrac = W / Wsum[:, None]
        Wproj = Wfrac.real
        Wproj[Wproj < 0] = 0

        W = Wsum[:, None] * Wproj

        return W


class BogusExtraFIDConstraint(FIDConstraint):

    def project_W(self, W, copy=False):
        """The phase of each component should be similar

        TODO: have some way of parametrising the permissible variation"""
        angle_limit = np.pi / 4
        angle_limit_wrap = 2 * np.pi - angle_limit

#        # Get the average phase for each sample and overall
#        sample_sum = np.sum(W, axis=1)
#        sample_phase = sample_sum / np.abs(sample_sum)
#        Wsum = np.sum(sample_sum)
#        Wphase = Wsum / np.abs(Wsum)
#        
#        # each sample's relative phase

#        Wsum = np.sum(W)
#        Wfrac = W / Wsum
#        Warg = np.angle(Wfrac)
#        Wabs = np.abs(Wfrac)
#        wWabs = Wabs * np.minimum(1, (1 + np.cos(Warg)) /
#                                     (1 + np.cos(angle_limit)))
#
#        arg_too_high = np.logical_and(Warg > angle_limit, Warg <= np.pi)
#        arg_too_low = np.logical_and(Warg > np.pi,  Warg < angle_limit_wrap)
#        W[arg_too_high] = Wsum * wWabs[arg_too_high] * np.exp(1j * angle_limit)
#        W[arg_too_low] = Wsum * wWabs[arg_too_low] * np.exp(1j * angle_limit_wrap)

        Wsum = np.sum(W, axis=1)
        Wfrac = W / Wsum[:, None]
        projWfrac = Wfrac.real
#        pwfmax = np.max(projWfrac, axis=0)
#        pwfmin = np.min(projWfrac, axis=0)
#        pwfmin *= (pwfmin < 0)
#        projWfrac = (projWfrac - pwfmin) * pwfmax / (pwfmax - pwfmin)
        
#        # Make sure the median is at least 0
#        pwfmed = np.median(projWfrac, axis=0)
#        pwfmed[pwfmed > 0] = 0
#        projWfrac += pwfmed[None, :]
#        projWfrac[projWfrac < 0] = 0
#        W = Wsum[:, None] * projWfrac
        
#        Wabs = np.abs(W)
#        Wsumabs = np.abs(Wsum)
#        W = (W + Wsum[:, None]) * Wabs / (Wabs + Wsumabs[:, None])

        return W


def svd_initialise(X, n_components, constraint):
    """Calculate a starting point for the generalised NMF fit

    The algorithm is based on the NNDSVD initialisation procedure"""
    U, S, V = linalg.svd(X, full_matrices=False)
    W = U[:, :n_components] * np.sqrt(S[None, :n_components])
    H = V[:n_components, :] * np.sqrt(S[:n_components, None])

    # Pick the best out of the positive and negative parts of each component
    # (This is NMF specific - TODO work out a more general approach)
    Wp = constraint.project_W(W, copy=True)
    Wm = constraint.project_W(-W, copy=True)
    Hp = constraint.project_H(H, copy=True)
    Hm = constraint.project_H(-H, copy=True)
    for i in range(n_components):
        pnorm = linalg.norm(Wp[:, i]) * linalg.norm(Hp[i, :])
        mnorm = linalg.norm(Wm[:, i]) * linalg.norm(Hm[i, :])
        if pnorm >= mnorm:
            W[:, i] = Wp[:, i]
            H[i, :] = Hp[i, :]
        else:
            W[:, i] = Wm[:, i]
            H[i, :] = Hm[i, :]

    W, H = constraint.normalise(W, H)
    return W, H


def projgrad_subproblem(V, W, H, project, sigma=0.01, beta=0.1):
    """Non-negative least square solver

    Solves a non-negative least squares subproblem using the
    projected gradient descent algorithm.
    min || WH - V ||_2

    Parameters
    ----------
    V, W : array-like
        Constant matrices.

    H : array-like
        Initial guess for the solution.

    project : function object
        Projects arbitrary H onto a valid value of H

    sigma : float
        Constant used in the sufficient decrease condition checked by the line
        search.  Smaller values lead to a looser sufficient decrease condition,
        thus reducing the time taken by the line search, but potentially
        increasing the number of iterations of the projected gradient
        procedure. 0.01 is a commonly used value in the optimization
        literature.

    beta : float
        Factor by which the step size is decreased (resp. increased) until
        (resp. as long as) the sufficient decrease condition is satisfied.
        Larger values allow to find a better step size but lead to longer line
        search. 0.1 is a commonly used value in the optimization literature.

    Returns
    -------
    H : array-like
        Solution to the non-negative least squares problem.

    grad : array-like
        The gradient.

    Reference
    ---------

    C.-J. Lin. Projected gradient methods
    for non-negative matrix factorization. Neural
    Computation, 19(2007), 2756-2779.
    http://www.csie.ntu.edu.tw/~cjlin/nmf/

    """
    WtV = np.dot(W.T, V)
    WtW = np.dot(W.T, W)

    # values justified in the paper
    alpha = 1

    # The convergence criterion for the outer iteration isn't so well defined
    # so we only do one iteration of the projected gradient line search here
    grad = np.dot(WtW, H) - WtV

    Hp = H

    for inner_iter in range(19):
        # Gradient step.
        Hn = H - alpha * grad
        # Projection step.
        Hn = project(Hn)
        d = Hn - H
        gradd = np.dot(grad.ravel(), d.ravel())
        dQd = np.dot(np.dot(WtW, d).ravel(), d.ravel())
        suff_decr = (1 - sigma) * gradd + 0.5 * dQd < 0
        if inner_iter == 0:
            decr_alpha = not suff_decr

        if decr_alpha:
            if suff_decr:
                H = Hn
                break
            else:
                alpha *= beta
        elif not suff_decr or (Hp == Hn).all():
            H = Hp
            break
        else:
            alpha /= beta
            Hp = Hn

    return H, grad


class BaseMF(object):

    def __init__(self, n_components, constraint, tol=1e-4, max_iter=200,
                 verbose=False):
        self.n_components = n_components
        self.constraint = constraint
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose

    # TODO create separate mixin classes allowing different initialisation
    # strategies
    def _init(self, X):
        W, H = svd_initialise(X, self.n_components, self.constraint)
        return W, H

    def fit_transform(self, X, y=None):
        """Learn a NMF model for the data X and returns the transformed data.

        This is more efficient than calling fit followed by transform.

        Parameters
        ----------

        X: array, shape = [n_samples, n_features]
            Data matrix to be decomposed

        Returns
        -------
        data: array, [n_samples, n_components]
            Transformed data
        """

        n_samples, n_features = X.shape

        W, H = self._init(X)

        Xnorm = linalg.norm(X)
        old_error = linalg.norm(X - np.dot(W, H))

        for n_iter in range(1, self.max_iter + 1):

            # update W
            W, gradW = self._update_W(X, H, W)

            # update H
            H, gradH = self._update_H(X, H, W)

            # Adjust the normalisation of W and H
            W, H = self.constraint.normalise(W, H)

            error = linalg.norm(X - np.dot(W, H))
            if self.verbose:
                print(n_iter, old_error, error)
            if error < self.tol * Xnorm:
                break
            elif error > old_error:
                warnings.warn("Error is getting worse")
                break
            elif error > old_error * (1 - self.tol):
                # Error is decreasing very slowly
                break
            old_error = error

        self.reconstruction_err_ = error
        self.components_ = H
        return W

    def fit(self, X, y=None, **params):
        """Learn a NMF model for the data X.

        Parameters
        ----------

        X: {array-like, sparse matrix}, shape = [n_samples, n_features]
            Data matrix to be decomposed

        Returns
        -------
        self
        """
        self.fit_transform(X, **params)
        return self


class ProjectedGradientNMF(BaseMF):
    """Non-Negative matrix factorization by Projected Gradient (NMF)

    Parameters
    ----------
    n_components : int
        Number of components, if n_components is not set all components
        are kept

    constraint : instance of a Constraint subclass
        Constraint on possible values of W and H

    tol : double, default: 1e-4
        Tolerance value used in stopping conditions.

    max_iter : int, default: 200
        Number of iterations to compute.

    Attributes
    ----------
    `components_` : array, [n_components, n_features]
        Non-negative components of the data.

    `reconstruction_err_` : number
        Frobenius norm of the matrix difference between
        the training data and the reconstructed data from
        the fit produced by the model. ``|| X - WH ||_2``

    References
    ----------
    This implements

    C.-J. Lin. Projected gradient methods
    for non-negative matrix factorization. Neural
    Computation, 19(2007), 2756-2779.
    http://www.csie.ntu.edu.tw/~cjlin/nmf/

    P. Hoyer. Non-negative Matrix Factorization with
    Sparseness Constraints. Journal of Machine Learning
    Research 2004.

    NNDSVD is introduced in

    C. Boutsidis, E. Gallopoulos: SVD based
    initialization: A head start for nonnegative
    matrix factorization - Pattern Recognition, 2008
    http://tinyurl.com/nndsvd
    """

    def _update_W(self, X, H, W):
        def project_WT(WT):
            W = self.constraint.project_W(WT.T)
            return W.T

        WT, gradWT = projgrad_subproblem(X.T, H.T, W.T, project_WT)
        return WT.T, gradWT.T

    def _update_H(self, X, H, W):
        H, gradH = projgrad_subproblem(X, W, H, self.constraint.project_H)
        return H, gradH


class PinvNMF(BaseMF):
    """Uses pseudo inverse rather than projected gradient to do the updates"""

    def _update_W(self, X, H, W_old, beta=0.5):
        # TODO find a faster way than calculating the norm of X - WH every time
        old_score = linalg.norm(X - np.dot(W_old, H))

        Hinv = linalg.pinv(H)
        W_ideal = np.dot(X, Hinv)
        Wdiff = W_ideal - W_old

        # Line search for an W that improves the score
        for i in range(10):
            alpha = beta ** i
            W = self.constraint.project_W(W_old + alpha * Wdiff)
            new_score = linalg.norm(X - np.dot(W, H))
            if self.verbose:
                print('W', i, old_score, new_score)
            if new_score <= old_score:
                break
        return W, Wdiff

    def _update_H(self, X, H_old, W, beta=0.5):
        # TODO find a faster way than calculating the norm of X - WH every time
        old_score = linalg.norm(X - np.dot(W, H_old))

        Winv = linalg.pinv(W)
        H_ideal = np.dot(Winv, X)
        Hdiff = H_ideal - H_old

        # Line search for an H that improves the score
        for i in range(10):
            alpha = beta ** i
            H = self.constraint.project_H(H_old + alpha * Hdiff)
            new_score = linalg.norm(X - np.dot(W, H))
            if self.verbose:
                print('H', i, old_score, new_score)
            if new_score <= old_score:
                break
        return H, Hdiff


class NMF(ProjectedGradientNMF):
    def __init__(self, n_components, tol=1e-4, max_iter=200):
        constraint = NMFConstraint()
        super(NMF, self).__init__(n_components, constraint, tol, max_iter)
