import unittest

import numpy as np
from scipy.sparse import coo_array, csr_array
import jax.numpy as jnp

from rafael.fedalgo import gwasprs

class SumUpTestCase(unittest.TestCase):

    def setUp(self):
        self.A = np.random.rand(2, 3, 4)
        self.B = np.random.rand(2, 3, 4)
        self.C = np.random.rand(2, 3, 4)

    def tearDown(self):
        self.A = None
        self.B = None
        self.C = None

    def test_sum_of_numpy_arrays(self):
        result = gwasprs.aggregations.SumUp()(self.A, self.B, self.C)
        ans = self.A + self.B + self.C
        np.testing.assert_array_almost_equal(ans, result)

    def test_sum_of_coo_array(self):
        result = gwasprs.aggregations.SumUp()(coo_array(self.A[0]), coo_array(self.B[0]), coo_array(self.C[0]))
        ans = self.A[0] + self.B[0] + self.C[0]
        np.testing.assert_array_almost_equal(ans, result.todense())

    def test_sum_of_csr_array(self):
        result = gwasprs.aggregations.SumUp()(csr_array(self.A[0]), csr_array(self.B[0]), csr_array(self.C[0]))
        ans = self.A[0] + self.B[0] + self.C[0]
        np.testing.assert_array_almost_equal(ans, result.todense())

    def test_sum_of_jax_arrays(self):
        result = gwasprs.aggregations.SumUp()(jnp.array(self.A), jnp.array(self.B), jnp.array(self.C))
        ans = jnp.array(self.A) + jnp.array(self.B) + jnp.array(self.C)
        np.testing.assert_array_almost_equal(ans, result)

    def test_sum_of_block_diags(self):
        A = gwasprs.block.BlockDiagonalMatrix([
            np.random.rand(3, 5),
            np.random.rand(7, 11),
            np.random.rand(13, 17),
        ])
        result = gwasprs.aggregations.SumUp()(A, A, A)
        ans = A.toarray() + A.toarray() + A.toarray()
        np.testing.assert_array_almost_equal(ans, result.toarray())

    def test_sum_of_numbers(self):
        result = gwasprs.aggregations.SumUp()(1, 2.5, 3.4)
        ans = 6.9
        self.assertEqual(ans, result)

    def test_sum_of_list_of_numpy_arrays(self):
        result = gwasprs.aggregations.SumUp()([self.A, self.B], [self.C, self.C])
        ans = [self.A + self.C, self.B + self.C]
        np.testing.assert_array_almost_equal(ans, result)

    def test_sum_of_list_of_jax_arrays(self):
        result = gwasprs.aggregations.SumUp()([jnp.array(self.A), jnp.array(self.B)], [jnp.array(self.C), jnp.array(self.C)])
        ans = [jnp.array(self.A + self.C), jnp.array(self.B + self.C)]
        np.testing.assert_array_almost_equal(ans, result)


class IntersectTestCase(unittest.TestCase):

    def setUp(self):
        self.A = np.array([2, 2, 5, 1, 7, 0, 10])
        self.B = np.array([1, 2, 3, 4, 4, 5, 7, 10, 0])
        self.C = np.array([0, 1, 2, 3, 4, 5, 6, 7, 10])

    def tearDown(self):
        self.A = None
        self.B = None
        self.C = None

    def test_intersect_of_list_of_list(self):
        result = gwasprs.aggregations.Intersect()(
            [2, 2, 5, 1, 7, 0, 10],
            [1, 2, 3, 4, 4, 5, 7, 10, 0],
            [0, 1, 2, 3, 4, 5, 6, 7, 10]
        )
        ans = [2, 5, 1, 7, 0, 10]
        self.assertEqual(ans, result)

    def test_intersect_of_numpy_arrays(self):
        result = gwasprs.aggregations.Intersect()(self.A, self.B, self.C)
        ans = np.array([2, 5, 1, 7, 0, 10])
        np.testing.assert_array_equal(ans, result)

    def test_intersect_of_jax_arrays(self):
        result = gwasprs.aggregations.Intersect()(jnp.array(self.A), jnp.array(self.B), jnp.array(self.C))
        ans = jnp.array([2, 5, 1, 7, 0, 10])
        np.testing.assert_array_equal(ans, result)

    def test_intersect_of_numbers(self):
        result = lambda : gwasprs.aggregations.Intersect()(1, 2.5, 3.4)
        self.assertRaises(NotImplementedError, result)

    def test_intersect_of_list_of_numpy_arrays(self):
        result = lambda : gwasprs.aggregations.Intersect()([self.A, self.B], [self.C, self.C])
        self.assertRaises(NotImplementedError, result)

    def test_intersect_of_list_of_jax_arrays(self):
        result = lambda : gwasprs.aggregations.Intersect()([jnp.array(self.A), jnp.array(self.B)], [jnp.array(self.C), jnp.array(self.C)])
        self.assertRaises(NotImplementedError, result)
