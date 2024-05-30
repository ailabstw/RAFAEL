import unittest
import numpy as np

from rafael.privacy import hyfed

class StrongRandomTestCase(unittest.TestCase):

    def setUp(self):
        self.size = (3, 4, 5)

    def tearDown(self):
        self.n = None

    def test_randint(self):
        low, high = 20, 80
        result = hyfed.randint(low, high, size=self.size)
        self.assertEqual(self.size, result.shape)
        self.assertEqual(np.int64, result.dtype)

    def test_rand(self):
        result = hyfed.rand(*self.size)
        self.assertEqual(self.size, result.shape)
        self.assertEqual(np.float64, result.dtype)

    def test_randn(self):
        result = hyfed.randn(*self.size)
        self.assertEqual(self.size, result.shape)
        self.assertEqual(np.float64, result.dtype)
