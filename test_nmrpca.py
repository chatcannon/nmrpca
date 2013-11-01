# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 19:13:53 2013

@author: chris
"""

import unittest

import numpy as np
from numpy.testing.utils import assert_array_equal

import nmrpca


class TestFlatten(unittest.TestCase):

    def test_single(self):
        carray = np.ones((2, 3), dtype='c8')

        rarray = nmrpca.nmr_flatten(carray)

        self.assertEqual((4, 3), tuple(rarray.shape))
        self.assertEqual(np.dtype('f4'), rarray.dtype)

    def test_double(self):
        carray = np.ones((2, 3), dtype='c16')

        rarray = nmrpca.nmr_flatten(carray)

        self.assertEqual((4, 3), tuple(rarray.shape))
        self.assertEqual(np.dtype('f8'), rarray.dtype)

    def test_flatten(self):
        carray = np.ones((2, 3, 4), dtype='c8')

        rarray = nmrpca.nmr_flatten(carray)

        self.assertEqual((4, 12), tuple(rarray.shape))

    def test_rebuild(self):
        realpart = np.random.rand(5, 4, 3)
        imagpart = np.random.rand(5, 4, 3)

        carray = realpart + 1j * imagpart

        rarray = nmrpca.nmr_flatten(carray)
        carray2 = nmrpca.nmr_rebuild(rarray)

        assert_array_equal(carray2, np.reshape(carray, (5, 12)))

if __name__ == '__main__':
    unittest.main()
