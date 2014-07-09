"""
Complex Principal Component Analysis
"""

# Author: Chris Kerr <cjk34@cam.ac.uk>
#
# This file contains some code copied from scikit-learn's implementation in
# the file sklearn/decomposition/pca.py, which has the following authors:
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Mathieu Blondel <mathieu@mblondel.org>
#         Denis A. Engemann <d.engemann@fz-juelich.de>
#
# License: BSD 3 clause

from math import sqrt

import numpy as np
from scipy import linalg


class PCA:
    """Complex PCA

    Replicates some of the interface of scikit-learn's PCA class"""

    def __init__(self, n_components=None, copy=True, whiten=False):
        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten

    def fit(self, X, y=None):
        """Fit the model with X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._fit(X)
        return self

    def fit_transform(self, X, y=None):
        """Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)

        """
        U, S, V = self._fit(X)
        U = U[:, :self.n_components_]

        if self.whiten:
            # X_new = X * V / S * sqrt(n_samples) = U * sqrt(n_samples)
            U *= sqrt(X.shape[0])
        else:
            # X_new = X * V = U * S * V^T * V = U * S
            U *= S[:self.n_components_]

        return U

    def _fit(self, X):
        """Fit the model on X

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        Returns
        -------
        U, s, V : ndarrays
            The SVD of the input data, copied and centered when
            requested.
        """
        n_samples, n_features = X.shape
        if self.copy:
            X = np.copy(X)
        # Center data
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_
        U, S, V = linalg.svd(X, full_matrices=False)
        explained_variance_ = (S ** 2) / n_samples
        explained_variance_ratio_ = (explained_variance_ /
                                     explained_variance_.sum())

        if self.whiten:
            components_ = V / (S[:, np.newaxis] / sqrt(n_samples))
        else:
            components_ = V

        n_components = self.n_components
        if n_components is None:
            n_components = n_features
        elif not 0 <= n_components <= n_features:
            raise ValueError("n_components=%r invalid for n_features=%d"
                             % (n_components, n_features))

        if 0 < n_components < 1.0:
            # number of components for which the cumulated explained variance
            # percentage is superior to the desired threshold
            ratio_cumsum = explained_variance_ratio_.cumsum()
            n_components = np.sum(ratio_cumsum < n_components) + 1

        # Compute noise covariance using Probabilistic PCA model
        # The sigma2 maximum likelihood (cf. eq. 12.46)
        if n_components < n_features:
            self.noise_variance_ = explained_variance_[n_components:].mean()
        else:
            self.noise_variance_ = 0.

        # store n_samples to revert whitening when getting covariance
        self.n_samples_ = n_samples

        self.components_ = components_[:n_components]
        self.explained_variance_ = explained_variance_[:n_components]
        explained_variance_ratio_ = explained_variance_ratio_[:n_components]
        self.explained_variance_ratio_ = explained_variance_ratio_
        self.n_components_ = n_components

        return (U, S, V)

    def transform(self, X):
        """Apply the dimensionality reduction on X.

        X is projected on the first principal components previous extracted
        from a training set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)

        """
        if self.mean_ is not None:
            X = X - self.mean_
        X_transformed = np.dot(X, np.conj(self.components_.T))
        return X_transformed

    def inverse_transform(self, X):
        """Transform data back to its original space, i.e.,
        return an input X_original whose transform would be X

        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            New data, where n_samples is the number of samples
            and n_components is the number of components.

        Returns
        -------
        X_original array-like, shape (n_samples, n_features)

        Notes
        -----
        If whitening is enabled, inverse_transform does not compute the
        exact inverse operation as transform.
        """
        return np.dot(X, self.components_) + self.mean_
