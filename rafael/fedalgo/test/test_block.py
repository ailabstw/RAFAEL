import unittest
import numpy as np
import jax.numpy as jnp

from rafael.fedalgo import gwasprs
from rafael.fedalgo.gwasprs import linalg
from rafael.fedalgo.gwasprs.block import block_diag


class BlockDiagonalTestCase(unittest.TestCase):

    def setUp(self):
        self.dim = 5
        self.Xs = [
            np.random.rand(7, self.dim),
            np.random.rand(11, self.dim),
            np.random.rand(13, self.dim),
        ]
        self.X = gwasprs.block.BlockDiagonalMatrix(self.Xs)
        self.Y = gwasprs.block.BlockDiagonalMatrix([
            np.random.rand(7, self.dim),
            np.random.rand(11, self.dim),
            np.random.rand(13, self.dim),
        ])
        self.y = np.random.rand(3 * self.dim)

    def tearDown(self):
        self.dim = None
        self.X = None
        self.y = None

    def test_ndim(self):
        self.assertEqual(2, self.X.ndim)

    def test_blocks(self):
        self.assertEqual(3, self.X.nblocks)
        self.assertEqual(self.X.nblocks, len(self.X.blocks))

        for (i, blk) in enumerate(self.X.blocks):
            np.testing.assert_array_almost_equal(self.Xs[i], blk, decimal=5)

        ans = [X.shape for X in self.Xs]
        self.assertEqual(ans, self.X.blockshapes)

        i = 1
        self.assertEqual(self.Xs[i].shape, self.X.blockshape(i))

    def test_append(self):
        result = gwasprs.block.BlockDiagonalMatrix(self.Xs[0:2])
        result.append(self.Xs[2])

        for (i, blk) in enumerate(result):
            np.testing.assert_array_almost_equal(self.Xs[i], blk, decimal=5)

    def test_append_block_diag(self):
        result = gwasprs.block.BlockDiagonalMatrix(self.Xs)
        Y = gwasprs.block.BlockDiagonalMatrix([
            np.random.rand(7, self.dim),
            np.random.rand(11, self.dim),
            np.random.rand(13, self.dim),
        ])
        result.append(Y)

        for (i, blk) in enumerate(self.Xs):
            np.testing.assert_array_almost_equal(blk, result[i], decimal=5)
        for (i, blk) in enumerate(Y):
            np.testing.assert_array_almost_equal(blk, result[i + len(self.Xs)], decimal=5)
            
    def test_fromlist(self):
        ls = [x.tolist() for x in self.Xs]
        result = gwasprs.block.BlockDiagonalMatrix.fromlist(ls)
        for (i, blk) in enumerate(result):
            np.testing.assert_array_equal(self.Xs[i], blk)
    
    def test_fromdense(self):
        n_block = 4
        Xs = [np.random.randn(10, self.dim) for _ in range(n_block)]
        result = gwasprs.block.BlockDiagonalMatrix.fromdense(block_diag(*Xs), n_block)
        for (i, blk) in enumerate(result):
            np.testing.assert_array_almost_equal(Xs[i], blk)
    
    def test_fromindex(self):
        indices = [
            [[0, 2],
             [0, 3],
             [0, 4]],
            [[2, 6],
             [3, 7],
             [4, 10]],
            [[6, 10],
             [7, 13],
             [10, 13]]
        ]
        Xs = []
        # Create matrices with unequal sizes
        for i in range(len(indices)):
            sizes = (end-start for start, end in indices[i])
            Xs.append(np.random.randn(*sizes))
        result = gwasprs.block.BlockDiagonalMatrix.fromindex(block_diag(*Xs), indices)
        for (i, blk) in enumerate(result):
            np.testing.assert_array_almost_equal(Xs[i], blk)

    def test_add(self):
        result = self.X + self.Y

        for (X, Y, res) in zip(self.X, self.Y, result):
            ans = X + Y
            np.testing.assert_array_almost_equal(ans, res, decimal=5)

    def test_mvdot(self):
        Xs = [
            np.random.rand(self.dim, 7),
            np.random.rand(self.dim, 11),
            np.random.rand(self.dim, 13),
        ]
        result = linalg.mvdot(gwasprs.block.BlockDiagonalMatrix(Xs), self.y)
        ans = np.concatenate([
            Xs[0].T @ self.y[0:self.dim],
            Xs[1].T @ self.y[self.dim:2*self.dim],
            Xs[2].T @ self.y[2*self.dim:3*self.dim],
        ])
        np.testing.assert_array_almost_equal(ans, result, decimal=5)

    def test_mvdot_vmap(self):
        Xs = [
            jnp.array(np.random.rand(self.dim, 7)),
            jnp.array(np.random.rand(self.dim, 11)),
            jnp.array(np.random.rand(self.dim, 13)),
        ]
        y = jnp.array(self.y)
        result = linalg.mvdot(gwasprs.block.BlockDiagonalMatrix(Xs), y)
        ans = np.concatenate([
            Xs[0].T @ self.y[0:self.dim],
            Xs[1].T @ self.y[self.dim:2*self.dim],
            Xs[2].T @ self.y[2*self.dim:3*self.dim],
        ])
        np.testing.assert_array_almost_equal(ans, result, decimal=5)

    def test_mvmul(self):
        result = linalg.mvmul(self.X, self.y)
        ans = np.concatenate([
            self.Xs[0] @ self.y[0:self.dim],
            self.Xs[1] @ self.y[self.dim:2*self.dim],
            self.Xs[2] @ self.y[2*self.dim:3*self.dim],
        ])
        np.testing.assert_array_almost_equal(ans, result, decimal=5)

    def test_mvmul_vmap(self):
        X = gwasprs.block.BlockDiagonalMatrix(list(map(jnp.array, self.Xs)))
        y = jnp.array(self.y)
        result = linalg.mvmul(X, y)
        ans = np.concatenate([
            self.Xs[0] @ self.y[0:self.dim],
            self.Xs[1] @ self.y[self.dim:2*self.dim],
            self.Xs[2] @ self.y[2*self.dim:3*self.dim],
        ])
        np.testing.assert_array_almost_equal(ans, result, decimal=5)

    def test_mvmul_multiprocess(self):
        result = linalg.mvmul(self.X, self.y, acceleration="process", n_jobs=2)
        ans = np.concatenate([
            self.Xs[0] @ self.y[0:self.dim],
            self.Xs[1] @ self.y[self.dim:2*self.dim],
            self.Xs[2] @ self.y[2*self.dim:3*self.dim],
        ])
        np.testing.assert_array_almost_equal(ans, result, decimal=5)

    def test_mmdot(self):
        result = linalg.mmdot(self.X, self.X)
        ans = [X.T @ X for X in self.Xs]

        for i in range(result.nblocks):
            np.testing.assert_array_almost_equal(ans[i], result[i], decimal=5)

    def test_mmdot_vmap(self):
        X = gwasprs.block.BlockDiagonalMatrix(list(map(jnp.array, self.Xs)))
        result = linalg.mmdot(X, X)
        ans = [X.T @ X for X in self.Xs]

        for i in range(result.nblocks):
            np.testing.assert_array_almost_equal(ans[i], result[i], decimal=5)

    def test_mmdot_multiprocess(self):
        result = linalg.mmdot(self.X, self.X, acceleration="process", n_jobs=2)
        ans = [X.T @ X for X in self.Xs]

        for i in range(result.nblocks):
            np.testing.assert_array_almost_equal(ans[i], result[i], decimal=5)

    def test_matmul(self):
        Ys = [
            np.random.rand(self.dim, 7),
            np.random.rand(self.dim, 11),
            np.random.rand(self.dim, 13),
        ]
        result = linalg.matmul(self.X, gwasprs.block.BlockDiagonalMatrix(Ys))
        ans = [X @ Y for (X, Y) in zip(self.Xs, Ys)]

        for i in range(result.nblocks):
            np.testing.assert_array_almost_equal(ans[i], result[i], decimal=5)

    def test_matmul_vmap(self):
        X = gwasprs.block.BlockDiagonalMatrix(list(map(jnp.array, self.Xs)))
        Ys = [
            jnp.array(np.random.rand(self.dim, 7)),
            jnp.array(np.random.rand(self.dim, 11)),
            jnp.array(np.random.rand(self.dim, 13)),
        ]
        result = linalg.matmul(X, gwasprs.block.BlockDiagonalMatrix(Ys))
        ans = [X @ Y for (X, Y) in zip(self.Xs, Ys)]

        for i in range(result.nblocks):
            np.testing.assert_array_almost_equal(ans[i], result[i], decimal=5)

    def test_matmul_multiprocess(self):
        Ys = [
            np.random.rand(self.dim, 7),
            np.random.rand(self.dim, 11),
            np.random.rand(self.dim, 13),
        ]
        result = linalg.matmul(self.X, gwasprs.block.BlockDiagonalMatrix(Ys), acceleration="process", n_jobs=2)
        ans = [X @ Y for (X, Y) in zip(self.Xs, Ys)]

        for i in range(result.nblocks):
            np.testing.assert_array_almost_equal(ans[i], result[i], decimal=5)

    def test_inv(self):
        cov = linalg.mmdot(self.X, self.X)
        result = linalg.inv(cov)

        for (i, blk) in enumerate(result):
            np.testing.assert_array_almost_equal(np.linalg.inv(cov[i]), blk, decimal=5)

class Testblockdiag(unittest.TestCase):
    def setUp(self):
        self.dim = 5
        self.Xs_2d = [
            np.random.randn(7, self.dim),
            np.random.randn(13, self.dim),
            np.random.randn(15, self.dim)
        ]
        self.Xs_3d = [
            np.random.randn(7, 7, self.dim),
            np.random.randn(13, 13, self.dim),
            np.random.randn(15, 15, self.dim)
        ]
    
    def test_blockdiag(self):
        result = block_diag(*self.Xs_2d)
        start = np.zeros(2, dtype='int')
        for x in self.Xs_2d:
            slices = tuple(slice(start[i], start[i]+x.shape[i]) for i in range(2))
            np.testing.assert_array_equal(x, result[slices])
            result[slices] = 0
            start += np.array(x.shape)
        np.testing.assert_array_equal(np.zeros(result.shape), result)
            
    def test_blockdiag_nd(self):
        result = block_diag(*self.Xs_3d)
        start = np.zeros(3, dtype='int')
        for x in self.Xs_3d:
            slices = tuple(slice(start[i], start[i] + x.shape[i]) for i in range(3))
            np.testing.assert_array_equal(x, result[slices])
            result[slices] = 0
            start += np.array(x.shape)
        np.testing.assert_array_equal(np.zeros(result.shape), result)
