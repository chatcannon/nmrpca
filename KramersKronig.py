# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 17:44:59 2013

@author: chris

This module contains classes for transforming real data into imaginary
data according to the Kramers-Kronig relationships.
"""

import numpy as np


class simpleKK:
    """A naive implementation of the Kramers-Kronig relationship"""

    def __init__(self):
        pass

    def imag(self, realdata):

        imagdata = np.zeros_like(realdata)
        npts = realdata.shape[0]

        for i in range(npts):
            for j in range(npts):
                if (i != j):
                    imagdata[i] += realdata[j] / (np.pi * (i-j))

        return imagdata

    def __call__(self, realdata):
        return realdata + 1j * self.imag(realdata)
