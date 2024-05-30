import os
os.environ['XLA_FLAGS'] = "--xla_force_host_platform_device_count=2"

import unittest

import numpy as np
import jax
import jax.numpy as jnp
from scipy.stats import norm
from scipy.sparse import csr_array

from rafael.fedalgo import gwasprs
import rafael.fedalgo.gwasprs.linalg as linalg


class LinearRegressionTestCase(unittest.TestCase):

    def setUp(self):
        self.n = 100
        self.dim = 10
        self.X = np.random.rand(self.n, self.dim)
        self.beta = np.random.rand(self.dim)
        self.y = np.dot(self.X, self.beta) + np.random.rand(self.n)
        self.model = gwasprs.LinearRegression(self.beta)

    def tearDown(self):
        self.n = None
        self.dim = None
        self.model = None
        self.X = None
        self.y = None

    def test_predict(self):
        result = self.model.predict(self.X)
        np.testing.assert_array_almost_equal(np.dot(self.X, self.beta), result, decimal=5)

    def test_sparse_predict(self):
        result = self.model.predict(csr_array(self.X))
        np.testing.assert_array_almost_equal(np.dot(self.X, self.beta), result, decimal=5)

    def test_fit(self):
        model = gwasprs.LinearRegression.fit(self.X, self.y)
        self.assertTrue(True)

    def test_residual(self):
        result = self.model.residual(self.X, self.y)
        np.testing.assert_array_almost_equal(self.y - self.model.predict(self.X), result, decimal=5)

    def test_sse(self):
        result = self.model.sse(self.X, self.y)
        resd = self.model.residual(self.X, self.y)
        np.testing.assert_array_almost_equal(np.vdot(resd.T, resd), result, decimal=5)


class LogisticRegressionTestCase(unittest.TestCase):

    def setUp(self):
        self.n = 100
        self.dim = 10
        self.X = np.random.rand(self.n, self.dim)
        self.beta = np.random.rand(self.dim)
        z = np.dot(self.X, self.beta)
        pred_y = norm.cdf(z - np.mean(z) + np.random.randn(self.n))
        binarize = lambda x: 1.0 if x > 0.5 else 0.0
        self.y = np.array(list(map(binarize, pred_y)))
        self.model = gwasprs.LogisticRegression(self.beta)

    def tearDown(self):
        self.n = None
        self.dim = None
        self.model = None
        self.X = None
        self.y = None

    def test_predict(self):
        result = self.model.predict(self.X)
        ans = 1 / (1 + np.exp(-np.dot(self.X, self.beta)))
        np.testing.assert_array_almost_equal(ans, result, decimal=5)

    def test_fit(self):
        self.model.fit(self.X, self.y)
        self.assertTrue(True)

    def test_residual(self):
        result = self.model.residual(self.X, self.y)
        ans = self.y - self.model.predict(self.X)
        np.testing.assert_array_almost_equal(ans, result, decimal=5)

    def test_gradient(self):
        result = self.model.gradient(self.X, self.y)
        ans = np.dot(self.X.T, self.model.residual(self.X, self.y))
        np.testing.assert_array_almost_equal(ans, result, decimal=5)

    def test_hessian(self):
        pass

    def test_loglikelihood(self):
        pass


class BatchedLinearRegressionTestCase(unittest.TestCase):

    def setUp(self):
        self.n = 111
        self.dim = 13
        self.batch_size = 3
        self.X = jnp.array(np.random.rand(self.batch_size, self.n, self.dim))
        self.beta = jnp.array(np.random.rand(self.batch_size, self.dim))
        self.y = linalg.batched_mvmul(self.X, self.beta) + jnp.array(np.random.rand(self.batch_size, self.n))
        self.model = gwasprs.regression.BatchedLinearRegression(self.beta)

    def tearDown(self):
        self.n = None
        self.dim = None
        self.model = None
        self.X = None
        self.y = None

    def test_predict(self):
        result = self.model.predict(self.X)
        np.testing.assert_array_almost_equal(linalg.batched_mvmul(self.X, self.beta), result, decimal=5)
        assert isinstance(result, jax.Array)

    def test_fit(self):
        model = gwasprs.regression.BatchedLinearRegression.fit(self.X, self.y, algo=linalg.BatchedInverseSolver())
        self.assertTrue(True)

    def test_residual(self):
        result = self.model.residual(self.X, self.y)
        np.testing.assert_array_almost_equal(self.y - self.model.predict(self.X), result, decimal=5)
        assert isinstance(result, jax.Array)

    def test_sse(self):
        result = self.model.sse(self.X, self.y)
        resd = self.model.residual(self.X, self.y)
        ans = linalg.batched_vdot(resd, resd)
        np.testing.assert_array_almost_equal(ans, result, decimal=5)
        assert isinstance(result, jax.Array)

    def test_dof(self):
        result = self.model.dof(jnp.array(np.repeat(self.n, [self.batch_size,])))
        ans = np.repeat(self.n - self.dim, [self.batch_size,])
        np.testing.assert_array_equal(ans, result)
        assert isinstance(result, jax.Array)

    def test_t_stats(self):
        sse = self.model.sse(self.X, self.y)
        XtX = linalg.batched_mmdot(self.X, self.X)
        dof = self.model.dof(np.repeat(self.n, [self.batch_size,]))
        result = self.model.t_stats(sse, XtX, dof)
        self.assertEqual((self.batch_size, self.dim), result.shape)
        assert isinstance(result, jax.Array)


class BatchedLogisticRegressionTestCase(unittest.TestCase):

    def setUp(self):
        self.n = 111
        self.dim = 13
        self.batch_size = 3
        self.X = jnp.array(np.random.rand(self.batch_size, self.n, self.dim))
        self.beta = jnp.array(np.random.rand(self.batch_size, self.dim))
        z = linalg.batched_mvmul(self.X, self.beta)
        pred_y = norm.cdf(z - np.mean(z) + np.random.randn(self.batch_size, self.n))
        binarize_2d = lambda batch: list(map(lambda x: 1 if x > 0.5 else 0, batch))
        self.y = jnp.array(list(map(binarize_2d, pred_y)))
        self.model = gwasprs.regression.BatchedLogisticRegression(self.beta)

    def tearDown(self):
        self.n = None
        self.dim = None
        self.model = None
        self.X = None
        self.y = None

    def test_predict(self):
        single_result = self.model.predict(self.X)
        self.model.acceleration = 'pmap'
        pmap_result = self.model.predict(self.X)
        predicted_y = 1 / (1 + np.exp(-linalg.batched_mvmul(self.X, self.beta)))
        np.testing.assert_array_almost_equal(predicted_y, single_result, decimal=5)
        np.testing.assert_array_almost_equal(predicted_y, pmap_result, decimal=5)
        assert isinstance(pmap_result, jax.Array)
        assert isinstance(single_result, jax.Array)

    def test_fit(self):
        self.model.fit(self.X, self.y)
        self.assertTrue(True)

    def test_residual(self):
        result = self.model.residual(self.X, self.y)
        ans = self.y - self.model.predict(self.X)
        np.testing.assert_array_almost_equal(ans, result, decimal=5)
        assert isinstance(result, jax.Array)

    def test_gradient(self):
        result = self.model.gradient(self.X, self.y)
        ans = linalg.batched_mvdot(self.X, self.model.residual(self.X, self.y))
        np.testing.assert_array_almost_equal(ans, result, decimal=5)
        assert isinstance(result, jax.Array)

    def test_hessian(self):
        result = self.model.hessian(self.X)
        assert isinstance(result, jax.Array)

    def test_loglikelihood(self):
        result = self.model.loglikelihood(self.X, self.y)
        assert isinstance(result, jax.Array)

    def test_inv_hessian(self):
        result = linalg.batched_inv(self.model.hessian(self.X))
        assert isinstance(result, jax.Array)


class BlockedLinearRegressionTestCase(unittest.TestCase):

    def setUp(self):
        self.n = 111
        self.dim = 3
        self.nmodels = 3
        self.X = gwasprs.block.BlockDiagonalMatrix([
            np.random.rand(self.n-1, self.dim),
            np.random.rand(self.n-10, self.dim),
            np.random.rand(self.n-100, self.dim),
        ])
        self.beta = np.random.rand(self.nmodels * self.dim)
        self.y = self.X @ self.beta + np.random.rand((self.n-1) + (self.n-10) + (self.n-100))
        self.model = gwasprs.regression.BlockedLinearRegression(self.beta, nmodels=self.nmodels)
        self.nobss = np.array([self.n-1, self.n-10, self.n-100])

    def tearDown(self):
        self.n = None
        self.dim = None
        self.model = None
        self.X = None
        self.y = None

    def test_predict(self):
        result = self.model.predict(self.X)
        np.testing.assert_array_almost_equal(self.X @ self.beta, result, decimal=5)

    def test_nmodels(self):
        self.assertEqual(self.nmodels, self.model.nmodels)

    def test_fit(self):
        model = gwasprs.regression.BlockedLinearRegression.fit(self.X, self.y, nmodels=self.nmodels)
        self.assertEqual(gwasprs.regression.BlockedLinearRegression, type(model))

    def test_residual(self):
        result = self.model.residual(self.X, self.y)
        np.testing.assert_array_almost_equal(self.y - self.model.predict(self.X), result, decimal=5)

    def test_sse(self):
        result = self.model.sse(self.X, self.y)
        resd = self.model.residual(self.X, self.y)
        ans = np.array([
            resd[0:self.n-1].T @ resd[0:self.n-1],
            resd[self.n-1:2*self.n-11].T @ resd[self.n-1:2*self.n-11],
            resd[2*self.n-11:3*self.n-111].T @ resd[2*self.n-11:3*self.n-111],
        ])
        np.testing.assert_array_almost_equal(ans, result, decimal=5)

    def test_dof(self):
        result = self.model.dof(self.nobss)
        ans = self.nobss - self.model.coef_dim
        np.testing.assert_array_equal(ans, result)

    def test_t_stats(self):
        sse = np.repeat(self.model.sse(self.X, self.y), [self.nmodels,])
        XtX = linalg.mmdot(self.X, self.X)
        dof = np.repeat(self.model.dof(self.nobss), [self.nmodels,])
        result = self.model.t_stats(sse, XtX, dof)
        self.assertEqual((self.nmodels * self.dim, ), result.shape)


class UtilsTestCase(unittest.TestCase):

    def setUp(self):
        self.n = 111
        self.dim = 13
        self.batch_size = 3
        self.X = np.random.rand(self.batch_size, self.n, self.dim)

    def tearDown(self):
        self.n = None
        self.dim = None
        self.model = None
        self.X = None

    def test_add_bias(self):
        Y = gwasprs.regression.add_bias(self.X, axis=2)
        self.assertEqual((self.batch_size, self.n, self.dim+1), Y.shape)
