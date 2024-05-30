import unittest
import numpy as np

from rafael.fedalgo.gwasprs import array 

class ArrayIteratorTestCase(unittest.TestCase):

    def setUp(self):
        self.A = np.array(
            [[1,2,3],
             [4,5,6],
             [7,8,9],
             [10, 11, 12]]
        )

    def tearDown(self):
        self.A = None

    def test_iter_over_rows(self):
        iter = array.ArrayIterator(self.A)

        for i in range(self.A.shape[0]):
            ans = np.expand_dims(self.A[i], 0)
            result = next(iter)
            np.testing.assert_array_equal(ans, result)

    def test_iter_over_cols(self):
        iter = array.ArrayIterator(self.A, axis=1)

        for i in range(self.A.shape[1]):
            ans = np.expand_dims(self.A[:, i], -1)
            result = next(iter)
            np.testing.assert_array_equal(ans, result)
