import unittest
from jax import jit, vmap
import numpy as np
from jax import numpy as jnp
from jax import scipy as jsp

from rafael.fedalgo import gwasprs


class CovarianceTestCase(unittest.TestCase):
    def setUp(self):
        self.d1 = 100
        self.d2 = 10
        self.X = np.random.rand(self.d1, self.d2)
        self.y = np.random.rand(self.d1)

    def tearDown(self):
        self.d1 = None
        self.d2 = None
        self.X = None
        self.y = None

    def test_unnorm_autocovariance(self):
        result = gwasprs.unnorm_autocovariance(self.X)
        ans = self.X.T @ self.X
        np.testing.assert_array_almost_equal(ans, result, decimal=5)

    def test_unnorm_covariance(self):
        result = gwasprs.unnorm_covariance(self.X, self.y)
        ans = self.X.T @ self.y
        np.testing.assert_array_almost_equal(ans, result, decimal=5)


class BlockCovarianceTestCase(unittest.TestCase):
    def setUp(self):
        self.d1 = 100
        self.d2 = 10
        self.A = np.random.rand(self.d1 - 1, self.d2)
        self.B = np.random.rand(self.d1 - 2, self.d2)
        self.C = np.random.rand(self.d1 - 3, self.d2)
        self.X = gwasprs.block.BlockDiagonalMatrix([self.A, self.B, self.C])
        self.y = np.random.rand(3 * self.d2)

    def tearDown(self):
        self.A = None
        self.B = None
        self.C = None
        self.X = None
        self.y = None

    def test_blocked_unnorm_autocovariance(self):
        result = gwasprs.stats.blocked_unnorm_autocovariance(self.X).toarray()
        ans = gwasprs.linalg.mmdot(self.X, self.X).toarray()
        np.testing.assert_array_almost_equal(ans, result, decimal=5)

    def test_blocked_unnorm_covariance(self):
        X = gwasprs.block.BlockDiagonalMatrix([self.A.T, self.B.T, self.C.T])
        result = gwasprs.stats.blocked_unnorm_covariance(X, self.y)
        ans = gwasprs.linalg.mvdot(X, self.y)
        np.testing.assert_array_almost_equal(ans, result, decimal=5)


@jit
def _logistic_stats(beta, inv_hessian):
    std = jnp.sqrt(inv_hessian.diagonal())
    t_stat = beta / std
    p_value = 1 - jsp.stats.chi2.cdf(jnp.square(t_stat), 1)
    return t_stat, p_value


@jit
def batched_logistic_stats_old(beta, inv_hessian):
    return vmap(_logistic_stats, (0, 0), (0, 0))(beta, inv_hessian)


class PvalueTestCase(unittest.TestCase):
    def setUp(self):
        self.beta = np.random.uniform(size=(10, 4))
        self.inv = np.random.uniform(size=(10, 4, 4))
        self.large_t = 1e60

    def tearDown(self):
        self.beta = None
        self.inv = None
        self.large_t = None

    def test_compare_to_old(self):
        t_stat0, p_value0 = gwasprs.stats.batched_logistic_stats(self.beta, self.inv)
        t_stat1, p_value1 = batched_logistic_stats_old(self.beta, self.inv)

        assert np.allclose(t_stat0, t_stat1)
        assert np.allclose(p_value0, p_value1)

    def test_low_pvalue(self):
        pvalue = gwasprs.stats.t_dist_pvalue(self.large_t, 4)
        assert pvalue > 0.0
