# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 18:28:45 2013

@author: chris
"""

import unittest

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from .KramersKronig import SimpleKK


class TestSimpleKK(unittest.TestCase):

    def setUp(self):
        self.KK = SimpleKK()

    def test_delta(self):
        realdata = np.asarray([0, 0, 1, 0, 0], dtype='f8')

        imagdata = self.KK.imag(realdata)

        assert_allclose(imagdata, np.array([-0.5, -1, 0, 1, 0.5]) / np.pi)
