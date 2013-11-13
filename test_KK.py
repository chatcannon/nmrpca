# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 18:28:45 2013

@author: chris
"""

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from .KramersKronig import SimpleKK, MatrixKK, StrideTricksMatrixKK


def test_SimpleKK_delta():
    sKK = SimpleKK()

    realdata = np.asarray([0, 0, 1, 0, 0], dtype='f8')

    imagdata = sKK.imag(realdata)

    assert_allclose(imagdata, np.array([0.5, 1, 0, -1, -0.5]) / np.pi)
    assert_array_equal(realdata + 1j * imagdata, sKK(realdata))


def test_MatrixKK_equals_SimpleKK(n=20):

    sKK = SimpleKK()
    mKK = MatrixKK(n)

    realdata = np.random.rand(n)

    assert_allclose(sKK(realdata), mKK(realdata))


def test_StrideTricksMatrix_equals_Matrix(n=100):

    mKK = MatrixKK(n)
    stKK = StrideTricksMatrixKK(n)

    assert_array_equal(mKK._matrix, stKK._matrix)
