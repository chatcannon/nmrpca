# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 17:50:43 2013

@author: cjk34
"""

import numpy as np

def nmr_flatten(nmrdata):
    """Convert an array of NMR data to a form suitable for PCA analysis

    Converts complex values to a (real, imag) pair of real numbers
    Flattens array dimensions greater than 2 onto the second dimension

    Arguments:
    nmrdata -- the array to flatten

    Returns:
    the flattened array
    """

    # Convert the complex NMR data into a 'real' array twice the size
    # Just do it the simple way for now, there is a way to do it in place
    nmrdata = np.concatenate(nmrdata.real, nmrdata.imag, axis=0)

    # Flatten the second and subsequent dimensions of the NMR data
    nmrdata = np.reshape(nmrdata, (nmrdata.shape[0],
                                   np.prod(nmrdata.shape[1:])))


def nmr_rebuild(nmrdata):
    """Undo the actions of nmr_flatten

    This converts the components returned from PCA into complex
    data suitable for further processing e.g. by Fourier Transform
    """

    npts = nmrdata.shape[0] / 2
    nmrdata = nmrdata[:npts, :] + 1j * nmrdata[npts:, :]
