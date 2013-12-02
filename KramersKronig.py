# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 17:44:59 2013

@author: chris

This module contains classes for transforming real data into imaginary
data according to the Kramers-Kronig relationships.
"""

from __future__ import division

import numpy as np
from numpy.dual import fft, ifft
from numpy.fft import fftshift
from numpy.lib.stride_tricks import as_strided
from scipy.signal import fftconvolve


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
                    imagdata[i] += realdata[j] / (np.pi * (j - i))

        return imagdata

    def __call__(self, realdata):
        """Return the full complex data corresponding to the real input"""
        return realdata + 1j * self.imag(realdata)


class MatrixKK:
    """Implementing the Kramers-Kronig calculation by matrix multiplication"""

    def __init__(self, N):
        c, r = np.ogrid[0:N, 0:N]  # row and column range vectors
        diaggrid = c - r  # NxN grid, kth diagonal = -k
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
        flat = -1j * np.pi * np.arange((1 - N), N)
        flat[N - 1] = 1
        flat = 1 / flat

        dtsz = flat.dtype.itemsize
        self._matrix = np.flipud(as_strided(flat, shape=(N, N),
                                            strides=(dtsz, dtsz)))


class ConvolveKK(SimpleKK):

    def __init__(self, N):
        self.N = N
        quarter_vec = (1.0 / (np.pi * np.arange(N - 1, N//2, -1)) -
                       1.0 / (np.pi * np.arange(1, N//2)))

        self.conv_vec = np.zeros((2*N - 1,))
        self.conv_vec[:N//2 - 1] = quarter_vec
        self.conv_vec[N//2:N - 1] = -quarter_vec[::-1]
        self.conv_vec[N:3*N//2 - 1] = quarter_vec
        self.conv_vec[3*N//2:] = -quarter_vec[::-1]

    def imag(self, realdata):
        assert (self.N,) == realdata.shape
        return fftconvolve(self.conv_vec, realdata, mode='valid')


class fftKK(SimpleKK):

    def imag(self, realdata):
        N = realdata.shape[0]
        tweak_fid = np.linspace(-1j, 1j, N, endpoint=False)
        tweak_fid[0] = 0
        fid_guess = ifft(fftshift(realdata))
        return fftshift(fft(fid_guess * tweak_fid)).real

    def __call__(self, realdata):
        N = realdata.shape[0]
        ## Not sure which of these ways of calculating tweak_fid is best
        ## TODO: test them out more thoroughly
        tweak_fid = np.empty((N,), dtype=float)
        tweak_fid[0] = 1
        tweak_fid[1:] = np.linspace(2, 0, N - 1, endpoint=True)        
        #tweak_fid = np.linspace(2, 0, N, endpoint=False)
        #tweak_fid[0] = 1
        fid_guess = ifft(fftshift(realdata))
        return fftshift(fft(fid_guess * tweak_fid))
