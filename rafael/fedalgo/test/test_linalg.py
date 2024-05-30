import unittest

import numpy as np
import scipy.linalg as slinalg
from jax import random
import jax.numpy as jnp
import jax.scipy as jsp

from rafael.fedalgo import gwasprs


class LinAlgTestCase(unittest.TestCase):

    def setUp(self):
        A = np.random.rand(4, 4)
        self.A = A.T @ A
        self.X = np.array(
            [[1,1,1],
             [2,2,2],
             [3,3,3],
             [4,4,4]]
        )
        self.y = np.array(
            [[1],
             [2],
             [3]]
        )

    def tearDown(self):
        self.X = None
        self.y = None

    def test_inv(self):
        A = self.A.copy()
        A[0, :] = 0
        A[:, 0] = 0
        result = gwasprs.linalg.inv(A)
        ans = np.linalg.inv(A[1:, 1:])
        np.testing.assert_array_almost_equal(ans, result[1:, 1:], decimal=5)

    def test_inverse_solver(self):
        y = np.random.randn(4)
        result = gwasprs.linalg.InverseSolver()(self.A, y)
        ans = slinalg.solve(self.A, y)
        np.testing.assert_array_almost_equal(ans, result, decimal=4)

    def test_cholesky_solver(self):
        y = np.random.randn(4)
        result = gwasprs.linalg.CholeskySolver()(self.A, y)
        ans = slinalg.solve(self.A, y)
        norm = np.linalg.norm(result - ans)
        self.assertAlmostEqual(norm, 0, places=5)

        # test for numerical stability
        A = self.A.copy()
        A[0, :] = 0
        A[:, 0] = 0
        result = gwasprs.linalg.CholeskySolver()(A, y)
        ans = slinalg.solve(A + np.finfo(A.dtype).eps, y)
        norm = np.linalg.norm(result - ans)
        self.assertAlmostEqual(norm, 0, places=5)

    def test_cholesky_solver_for_block_diagonal(self):
        A = gwasprs.block.BlockDiagonalMatrix([
            np.random.rand(13, 4),
            np.random.rand(17, 4),
            np.random.rand(19, 4),
        ])
        A = gwasprs.linalg.mmdot(A, A)
        y = np.random.randn(3*4)
        result = gwasprs.linalg.CholeskySolver()(A, y)
        ans = slinalg.solve(A.toarray(), y)
        norm = np.linalg.norm(result - ans)
        self.assertAlmostEqual(norm, 0, places=5)


class BatchedLinAlgTestCase(unittest.TestCase):

    def setUp(self):
        key = random.PRNGKey(758493)
        A = random.uniform(key, shape=(4, 4))
        A = jnp.expand_dims(A.T @ A, axis=0)
        X = np.array(
            [[[1,1,1],
              [2,2,2],
              [3,3,3],
              [4,4,4]]]
        )
        Y = np.array(
            [[[1, 1, 1],
              [2, 2, 2],
              [3, 3, 3]]]
        )
        y = np.array(
            [[[1],
              [2],
              [3]]]
        )
        self.A = np.concatenate((A, A), axis=0)
        self.X = np.concatenate((X, X), axis=0)
        self.Y = np.concatenate((Y, Y), axis=0)
        self.y = np.concatenate((y, y), axis=0)

    def tearDown(self):
        self.X = None
        self.Y = None
        self.y = None

    def test_batched_mvmul(self):
        result = gwasprs.linalg.batched_mvmul(self.X, self.y)
        ans = np.array(
            [[ 6, 12, 18, 24],
             [ 6, 12, 18, 24]]
        )
        np.testing.assert_array_equal(ans, result)

    def test_batched_matmul(self):
        result = gwasprs.linalg.batched_matmul(self.X, self.Y)
        ans = np.array(
            [[[ 6,  6,  6],
              [12, 12, 12],
              [18, 18, 18],
              [24, 24, 24]],

             [[ 6,  6,  6],
              [12, 12, 12],
              [18, 18, 18],
              [24, 24, 24]]]
        )
        np.testing.assert_array_equal(ans, result)

    def test_batched_mvdot(self):
        result = gwasprs.linalg.batched_mvdot(self.Y, self.y)
        ans = np.array(
            [[14, 14, 14],
             [14, 14, 14]]
        )
        np.testing.assert_array_equal(ans, result)

    def test_batched_mmdot(self):
        result = gwasprs.linalg.batched_mmdot(self.Y, self.Y)
        ans = np.array(
            [[[14, 14, 14],
              [14, 14, 14],
              [14, 14, 14]],

             [[14, 14, 14],
              [14, 14, 14],
              [14, 14, 14]]]
        )
        np.testing.assert_array_equal(ans, result)

    def test_batched_diagonal(self):
        d = np.array([1,2,3])
        D = np.diag(d)
        D = np.expand_dims(D, axis=0)
        D = np.concatenate((D, D), axis=0)
        result = gwasprs.linalg.batched_diagonal(D)

        d = np.expand_dims(d, axis=0)
        ans = np.concatenate((d, d), axis=0)
        np.testing.assert_array_equal(ans, result)

    def test_batched_cholesky(self):
        L = np.linalg.cholesky(self.A[0, :, :])
        result = gwasprs.linalg.batched_cholesky(self.A)

        L = np.expand_dims(L, axis=0)
        ans = np.concatenate((L, L), axis=0)
        np.testing.assert_array_equal(ans, result)

    def test_batched_cholesky_solver(self):
        y = np.random.randn(4)
        y = np.expand_dims(y, axis=0)
        y = np.concatenate((y, y), axis=0)
        result = gwasprs.linalg.BatchedCholeskySolver()(self.A, y)
        ans = slinalg.solve(self.A[0, :, :], y[0, :])
        ans = np.expand_dims(ans, axis=0)
        ans = np.concatenate((ans, ans), axis=0)
        norm = np.linalg.norm(result - ans)
        self.assertAlmostEqual(norm, 0, places=5)



class FederatedSVDTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        key = random.PRNGKey(758493)
        self.n_edges, self.n_samples, self.n_SNPs = 4, [30,50,40,60], 200
        self.k1, self.k2, self.max_iterations, self.epsilon = 20, 20, 20, 1e-9
        self.As = [gwasprs.array.simulate_genotype_matrix(key, shape=(self.n_samples[edge_idx], self.n_SNPs)) for edge_idx in range(self.n_edges)]
        split_idx = [sum(self.n_samples[:edge_idx+1]) for edge_idx in range(self.n_edges)]

        # Standardization answer
        global_A = np.concatenate(self.As, axis=0)
        mean = np.nanmean(global_A, axis=0)
        na_idx = np.where(np.isnan(global_A))
        global_A[na_idx] = np.take(mean, na_idx[1])
        self.global_A_ans = (global_A-np.mean(global_A,axis=0))/np.nanstd(global_A,axis=0,ddof=1)
        self.global_A_ans = np.delete(self.global_A_ans, np.isnan(self.global_A_ans[0]),axis=1)
        self.global_As_ans = np.vsplit(self.global_A_ans, split_idx)[:self.n_edges]

        # SVD answer
        self.global_G_ans, s, self.global_H_ans = jsp.linalg.svd(self.global_A_ans, full_matrices=False)
        self.global_G_ans = self.global_G_ans[:,0:self.k2]
        self.global_H_ans = self.global_H_ans.T[:,0:self.k2]

        # Orthonormalization init
        G, _ = gwasprs.array.simulate_eigenvectors(np.sum(self.n_samples), self.n_SNPs, self.k2)
        self.Gs = np.vsplit(G, split_idx)[:self.n_edges]

    def test_federated_standardization(self):
        result = np.concatenate(gwasprs.linalg.FederatedStandardization.standalone(self.As.copy()), axis=0)
        np.testing.assert_array_almost_equal(self.global_A_ans, result, decimal=5)

    def test_federated_vertical_subspace_iteration(self):
        As, Hs, local_Gs = gwasprs.linalg.FederatedVerticalSubspaceIteration.standalone(self.global_As_ans.copy(), self.k1, self.epsilon, self.max_iterations)
        for i in range(len(As)):
            assert As[i].T.shape == self.global_As_ans[i].shape
            assert Hs[i].shape == (self.n_SNPs, self.k2)
            assert local_Gs[i].shape == (self.n_samples[i], self.k2)

    def test_federated_randomized_svd(self):
        # Vertical subspace iterations, the output As (n_SNPs, n_samples)
        As, Hs, local_Gs = gwasprs.linalg.FederatedVerticalSubspaceIteration.standalone(self.global_As_ans.copy(), self.k1, self.epsilon, self.max_iterations)

        # Randomized SVD
        result_H, result_Gs = gwasprs.linalg.FederatedRandomizedSVD.standalone(As, Hs, local_Gs, self.k2)

        # Evaluations
        result_G = np.concatenate(result_Gs, axis=0)
        result_G = jsp.linalg.qr(result_G, mode='economic')[0]

        gwasprs.linalg.eigenvec_concordance_estimation(self.global_G_ans, result_G, decimal=3)
        gwasprs.linalg.eigenvec_concordance_estimation(self.global_H_ans, result_H, decimal=3)

    def test_federated_gram_schmidt_orthonormalization_(self):
        gwasprs.project.FederatedGramSchmidt.standalone(self.Gs)

    def test_federated_svd(self):
        # Federated SVD
        result_Gs, result_H = gwasprs.linalg.FederatedSVD.standalone(self.As.copy(), self.k1, self.k2, self.epsilon, self.max_iterations)

        # Evaluations
        result_G = np.concatenate(result_Gs, axis=0)

        gwasprs.linalg.eigenvec_concordance_estimation(self.global_G_ans, result_G, decimal=3)
        gwasprs.linalg.eigenvec_concordance_estimation(self.global_H_ans, result_H, decimal=3)





