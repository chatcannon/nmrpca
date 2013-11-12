# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 17:44:59 2013

@author: chris

This module contains classes for transforming real data into imaginary
data according to the Kramers-Kronig relationships.
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided


class SimpleKK:
    """A naive implementation of the Kramers-Kronig relationship"""

    def __init__(self):
        pass

    def imag(self, realdata):
        """Return the imaginary part corresponding to a given real part"""

        imagdata = np.zeros_like(realdata)
        npts = realdata.shape[0]

        for i in range(npts):
            for j in range(npts):
                if (i != j):
                    imagdata[i] += realdata[j] / (np.pi * (i-j))

        return imagdata

    def __call__(self, realdata):
        """Return the full complex data corresponding to the real input"""
        return realdata + 1j * self.imag(realdata)


class MatrixKK:
    """Implementing the Kramers-Kronig calculation by matrix multiplication"""

    def __init__(self, N):
        c, r = np.ogrid[0:N, 0:N]  # row and column range vectors
        diaggrid = r - c  # NxN grid, kth diagonal = k
        # N.B. 1 / diaggrid would give a divide-by-zero error
        # Using a complex grid with 1/1=1 real on the diagonal
        self._matrix = 1 / (np.eye(N) + 1j * np.pi * diaggrid)

    def imag(self, realdata):
        """Return the imaginary part corresponding to a given real part"""
        return np.dot(self._matrix.imag, realdata)

    def __call__(self, realdata):
        """Return the full complex data corresponding to the real input"""
        return np.dot(self._matrix, realdata)


class StrideTricksMatrixKK(MatrixKK):
    """Build the internal matrix in a compressed format using stride_tricks"""

    def __init__(self, N):
        flat = 1j * np.pi * np.arange((1 - N), N)
        flat[N - 1] = 1
        flat = 1 / flat

        dtsz = flat.dtype.itemsize
        self._matrix = np.flipud(as_strided(flat, shape=(N, N),
                                            strides=(dtsz, dtsz)))
