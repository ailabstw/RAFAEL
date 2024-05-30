import abc
import multiprocessing as mp

import numpy as np
import scipy.linalg as slinalg
import jax
from jax import jit, vmap
from jax import numpy as jnp
from jax import scipy as jsp
from jax import random as jrand

from . import stats, aggregations, block
from .project import FederatedGramSchmidt


def nansum(A):
    snp_sum = jnp.nansum(A, axis=0)
    non_na_count = jnp.count_nonzero(~jnp.isnan(A))
    return snp_sum, non_na_count


def mvdot(
    X: "np.ndarray[(1, 1), np.floating]", y: "np.ndarray[(1,), np.floating]"
) -> "np.ndarray[(1,), np.floating]":
    """Matrix-vector dot product

    Perform X.T * y.

    Args:
        X (np.ndarray[(1, 1), np.floating]): Matrix.
        y (np.ndarray[(1,), np.floating]): Vector.

    Returns:
        np.ndarray[(1,), np.floating]: Vector.
    """
    # assert X.ndim == 2 and y.ndim == 1
    if isinstance(X, block.BlockDiagonalMatrix):
        rowidx = np.cumsum([0] + [shape[0] for shape in X.blockshapes])
        colidx = np.cumsum([0] + [shape[1] for shape in X.blockshapes])
        res = np.empty(colidx[-1])
        for i in range(X.nblocks):
            res.view()[colidx[i] : colidx[i + 1]] = (
                X[i].T @ y.view()[rowidx[i] : rowidx[i + 1]]
            )
        return res
    else:
        # fallback
        if isinstance(X, jax.Array) or isinstance(y, jax.Array):
            return jit(vmap(jnp.vdot, (1, None), 0))(X, y)
        else:
            return X.T @ y


def mvmul(
    X: "np.ndarray[(1, 1), np.floating]",
    y: "np.ndarray[(1,), np.floating]",
    acceleration: str = "none",
    n_jobs: int = 1,
) -> "np.ndarray[(1,), np.floating]":
    """Matrix-vector multiplication

    Perform X * y.

    Args:
        X (np.ndarray[(1, 1), np.floating]): Matrix.
        y (np.ndarray[(1,), np.floating]): Vector.

    Returns:
        np.ndarray[(1,), np.floating]: Vector.
    """
    # assert X.ndim == 2 and y.ndim == 1
    if isinstance(X, block.AbstractBlockDiagonalMatrix):
        if acceleration == "process":
            pool = mp.Pool(n_jobs)
            results = pool.starmap_async(mvmul, zip(X.blocks, np.split(y, X.nblocks)))
            pool.close()
            pool.join()
            return np.concatenate(results.get())
        elif acceleration == "pmap":
            raise NotImplementedError("pmap acceleration is not implemented.")
        else:
            return X @ y
    else:
        # fallback
        if isinstance(X, jax.Array) or isinstance(y, jax.Array):
            return jit(vmap(jnp.vdot, (0, None), 0))(X, y)
        else:
            return X @ y


def mmdot(
    X: "np.ndarray[(1, 1), np.floating]",
    Y: "np.ndarray[(1, 1), np.floating]",
    acceleration: str = "none",
    n_jobs: int = 1,
) -> "np.ndarray[(1, 1), np.floating]":
    """Matrix-matrix dot product

    Perform X.T * Y.

    Args:
        X (np.ndarray[(1, 1), np.floating]): Matrix.
        Y (np.ndarray[(1, 1), np.floating]): Matrix.

    Returns:
        np.ndarray[(1, 1), np.floating]: Matrix.
    """
    assert X.ndim == Y.ndim == 2
    if isinstance(X, block.AbstractBlockDiagonalMatrix) and isinstance(
        Y, block.AbstractBlockDiagonalMatrix
    ):
        if acceleration == "process":
            pool = mp.Pool(n_jobs)
            results = pool.starmap_async(mmdot, zip(X.blocks, Y.blocks))
            pool.close()
            pool.join()
            return block.BlockDiagonalMatrix(results.get())
        elif acceleration == "pmap":
            raise NotImplementedError("pmap acceleration is not implemented.")
        else:
            return block.BlockDiagonalMatrix(
                [x.T @ y for (x, y) in zip(X.blocks, Y.blocks)]
            )
    else:
        # fallback
        if isinstance(X, jax.Array) or isinstance(Y, jax.Array):
            return jit(vmap(mvmul, (None, 1), 1))(X.T, Y)
        else:
            return X.T @ Y


def matmul(
    X: "np.ndarray[(1, 1), np.floating]",
    Y: "np.ndarray[(1, 1), np.floating]",
    acceleration: str = "none",
    n_jobs: int = 1,
) -> "np.ndarray[(1, 1), np.floating]":
    """Matrix multiplication

    Perform X * Y.

    Args:
        X (np.ndarray[(1, 1), np.floating]): Matrix.
        Y (np.ndarray[(1, 1), np.floating]): Matrix.

    Returns:
        np.ndarray[(1, 1), np.floating]: Matrix.
    """
    assert X.ndim == Y.ndim == 2
    if isinstance(X, block.AbstractBlockDiagonalMatrix) and isinstance(
        Y, block.AbstractBlockDiagonalMatrix
    ):
        if acceleration == "process":
            pool = mp.Pool(n_jobs)
            results = pool.starmap_async(matmul, zip(X.blocks, Y.blocks))
            pool.close()
            pool.join()
            return block.BlockDiagonalMatrix(results.get())
        elif acceleration == "pmap":
            raise NotImplementedError("pmap acceleration is not implemented.")
        else:
            return X @ Y
    else:
        # fallback
        if isinstance(X, jax.Array) or isinstance(Y, jax.Array):
            return jit(vmap(mvmul, (None, 1), 1))(X, Y)
        else:
            return X @ Y


def gen_mvmul(y: np.ndarray):
    @jit
    def _mvmul(X: np.ndarray) -> np.ndarray:
        return vmap(jnp.vdot, (0, None), 0)(X, y)

    return _mvmul


def inv(X):
    # Add machine eps to avoid zeros in matrix and increase numerical stability
    if isinstance(X, block.AbstractBlockDiagonalMatrix):
        return block.BlockDiagonalMatrix(
            [np.linalg.inv(blk + np.finfo(blk.dtype).eps) for blk in X.blocks]
        )
    else:
        # fallback
        return np.linalg.inv(X + np.finfo(X.dtype).eps)


def batched_vdot(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Batched vector-vector dot product

    Perform x.T * y with batch on their first dimension.

    Args:
        x (np.ndarray[(1, 1), np.floating]): Batched vector.
        y (np.ndarray[(1, 1), np.floating]): Batched vector.

    Returns:
        np.ndarray[(1, 1), np.floating]: Batched vector.
    """
    return jnp.sum(x * y, axis=1)


@jit
def batched_mvdot(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Batched matrix-vector dot product

    Perform X.T * y with batch on their first dimension.

    Args:
        X (np.ndarray[(1, 1, 1), np.floating]): Batched matrix.
        y (np.ndarray[(1, 1), np.floating]): Batched vector.

    Returns:
        np.ndarray[(1, 1), np.floating]: Batched vector.
    """
    return vmap(mvdot, (0, 0), 0)(X, y)


@jit
def batched_mvmul(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Batched matrix-vector multiplication

    Perform X * y with batch on their first dimension.

    Args:
        X (np.ndarray[(1, 1, 1), np.floating]): Batched matrix.
        y (np.ndarray[(1, 1), np.floating]): Batched vector.

    Returns:
        np.ndarray[(1, 1), np.floating]: Batched vector.
    """
    return vmap(mvmul, (0, 0), 0)(X, y)


@jit
def batched_mmdot(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Batched matrix-matrix dot product

    Perform X.T * Y with batch on their first dimension.

    Args:
        X (np.ndarray[(1, 1, 1), np.floating]): Batched matrix.
        Y (np.ndarray[(1, 1, 1), np.floating]): Batched matrix.

    Returns:
        np.ndarray[(1, 1, 1), np.floating]: Batched matrix.
    """
    return vmap(mmdot, 0, 0)(X, Y)


@jit
def batched_matmul(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Batched matrix multiplication

    Perform X * Y with batch on their first dimension.

    Args:
        X (np.ndarray[(1, 1, 1), np.floating]): Batched matrix.
        Y (np.ndarray[(1, 1, 1), np.floating]): Batched matrix.

    Returns:
        np.ndarray[(1, 1, 1), np.floating]: Batched matrix.
    """
    return vmap(matmul, 0, 0)(X, Y)


@jit
def batched_diagonal(X: np.ndarray) -> np.ndarray:
    return vmap(jnp.diagonal, 0, 0)(X)


@jit
def batched_inv(X: np.ndarray) -> np.ndarray:
    return vmap(jnp.linalg.inv, 0, 0)(X)


def batched_cholesky(X: np.ndarray) -> np.ndarray:
    batch_size = X.shape[0]
    L = np.empty(X.shape)
    for b in range(batch_size):
        L.view()[b, :, :] = np.linalg.cholesky(X[b, :, :])
    return L


@jit
def batched_solve(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return vmap(jsp.linalg.solve, (0, 0), 0)(X, y)


@jit
def batched_solve_lower_triangular(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return vmap(lambda X, y: jsp.linalg.solve_triangular(X, y, lower=True), (0, 0), 0)(
        X, y
    )


@jit
def batched_solve_trans_lower_triangular(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return vmap(
        lambda X, y: jsp.linalg.solve_triangular(X, y, trans="T", lower=True), (0, 0), 0
    )(X, y)


class LinearSolver(object, metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        pass


class InverseSolver(LinearSolver):
    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self, X: "np.ndarray[(1, 1), np.floating]", y: "np.ndarray[(1,), np.floating]"
    ):
        # solve beta for X @ beta = y
        # Add machine eps to avoid zeros in matrix and increase numerical stability
        return jnp.linalg.solve(X + np.finfo(X.dtype).eps, y)


class BatchedInverseSolver(LinearSolver):
    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self,
        X: "np.ndarray[(1, 1, 1), np.floating]",
        y: "np.ndarray[(1, 1), np.floating]",
    ):
        # solve beta for X @ beta = y
        return batched_solve(X, y)


class CholeskySolver(LinearSolver):
    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self, X: "np.ndarray[(1, 1), np.floating]", y: "np.ndarray[(1,), np.floating]"
    ):
        if isinstance(X, jax.Array):
            # L = Cholesky(X)
            # Add machine eps to avoid zeros in matrix and increase numerical stability
            L = jnp.linalg.cholesky(X + np.finfo(X.dtype).eps)
            # solve Lz = y
            z = jsp.linalg.solve_triangular(L, y, lower=True)
            # solve Lt beta = z
            return jsp.linalg.solve_triangular(L, z, trans="T", lower=True)
        elif isinstance(X, (np.ndarray, np.generic)):
            # Add machine eps to avoid zeros in matrix and increase numerical stability
            c, low = slinalg.cho_factor(X + np.finfo(X.dtype).eps)
            return slinalg.cho_solve((c, low), y)
        elif isinstance(X, block.BlockDiagonalMatrix):
            start = 0
            res = np.empty(X.shape[0])
            for A in X:
                d = A.shape[1]
                # Add machine eps to avoid zeros in matrix and increase numerical stability
                c, low = slinalg.cho_factor(A + np.finfo(A.dtype).eps)
                x = slinalg.cho_solve((c, low), y.view()[start : start + d])
                res.view()[start : start + d] = x
                start += d
            return res
        else:
            raise Exception(f"CholeskySolver doesn't support matrix of type {type(X)}")


class BatchedCholeskySolver(LinearSolver):
    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self,
        X: "np.ndarray[(1, 1, 1), np.floating]",
        y: "np.ndarray[(1, 1), np.floating]",
    ):
        # L = Cholesky(X)
        L = batched_cholesky(X)
        # solve Lz = y
        z = batched_solve_lower_triangular(L, y)
        # solve Lt beta = z
        return batched_solve_trans_lower_triangular(L, z)


class QRSolver(LinearSolver):
    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self, X: "np.ndarray[(1, 1), np.floating]", y: "np.ndarray[(1,), np.floating]"
    ):
        # Q, R = QR(X)
        Q, R = jnp.linalg.qr(X)
        # solve R beta = Qty
        return jsp.linalg.solve(R, mvdot(Q, y), lower=False)


@jit
def orthogonalize(v, ortho, res):
    """Orthogonalize

    v - (summation of v's projections on i-1 orthogonalized eigenvectors)

    Args:
        v (np.ndarray[(1,), np.floating]) : ith eigenvector to be orthogonalized
        ortho (list of np.ndarray[(1,), np.floating]): i-1 orthogonalized eigenvectors with shape (n,)
        res (np.ndarray[(1,), np.floating]): residuals with shape (i-1,)

    Returns:
        (np.ndarray[(1,), np.floating]) : ith orthogonalized eigenvector
    """
    ortho = jnp.asarray(ortho)
    res = jnp.expand_dims(jnp.array(res), -1)
    projection = jnp.sum(res * ortho, axis=0)
    return v - projection


@jit
def svd(X):
    return jsp.linalg.svd(X, full_matrices=False)


def randn(n, m, seed=42):
    return jrand.normal(key=jrand.PRNGKey(seed), shape=(n, m))


def check_eigenvector_convergence(current, previous, tolerance, required=None):
    """
    This function checks whether two sets of vectors are assymptotically collinear,
    up to a tolerance of epsilon.

    Args:
        current: The current eigenvector estimate
        previous: The eigenvector estimate from the previous iteration
        tolerance: The error tolerance for eigenvectors to be equal
        required: optional parameter for the number of eigenvectors required to have converged
    Returns: True if the required numbers of eigenvectors have converged to the given precision, False otherwise
                deltas, the current difference between the dot products
    """

    nr_converged = 0
    col = 0
    converged = False
    deltas = []
    if required is None:
        required = current.shape[1]
    while col < current.shape[1] and not converged:
        # check if the scalar product of the current and the previous eigenvectors
        # is 1, which means the vectors are 'parallel'
        delta = jnp.abs(
            jnp.sum(jnp.dot(jnp.transpose(current[:, col]), previous[:, col]))
        )
        deltas.append(delta)
        if delta >= 1 - tolerance:
            nr_converged = nr_converged + 1
        if nr_converged >= required:
            converged = True
        col = col + 1
    return converged, deltas


def update_Us(U, Us, current_iteration):
    Us.append(U)
    return U, Us, current_iteration + 1


def init_rand_U(m, k1):
    """Initial U matrix generation

    Generate random U matrix with shape (m, k1),
    where m and k1 represent m features and k1 latent dimensions respectively.

    Args:
        m (int) : number of features
        k1 (int) : latent dimensions of U matrix in SVD and must be <= n samples

    Returns:
        (np.ndarray[(1,1), np.floating]) : random U matrix
    """
    prev_U = randn(m, k1)
    return prev_U


def update_local_U(A, V):
    """Update U matrix in edge

    U = AV, where U (m, k1), A (m, n) and V (n, k1)

    Args:
        A (np.ndarray[(1,1), np.floating]) : matrix with shape (m, n), where m and n represent features and samples respectively.
        V (np.ndarray[(1,1), np.floating]) : randomly generated and orthonormalized V matrix in the 1st step or updated V matrix during iterations.

    Returns:
        (np.ndarray[(1,1), np.floating]) : updated U matrix (m, k1)
    """
    U = mmdot(A.T, V)
    return U


def orthonormalize(M):
    """Orthonormalize matrix in aggregator

    Algo2/10-11
    orthonormalization of H matrix collected from edges.

    Args:
        M (np.ndarray[(1,1), np.floating]) : aggregated matrix collected from edges.

    Return:
        (np.ndarray[(1,1), np.floating]) : orthonormalized matrix (m, k1)
        (np.ndarray[(1,), np.floating]) : singular values
    """
    M, S = jsp.linalg.qr(M, mode="economic")
    S = abs(jnp.diag(S))
    return M, S


def update_local_V(A, U):
    """Update V matrix in edge

    V = AtU, where V (n, k1), At (n, m) and U (m, k1)

    Args:
        A (np.ndarray[(1,1), np.floating]) : matrix with shape (m, n), where m and n represent m features and n samples respectively.
        H (np.ndarray[(1,1), np.floating]) : global U matrix from aggregator with shape (m, k1)

    Returns:
        (np.ndarray[(1,1), np.floating]) : update V matrix with shape (n, k1)
    """
    V = mmdot(A, U)
    return V


def decompose_U_stack(Us):
    """Stack U matrices from I iterations and decompose it

    Each U matrix is the updated U matrix during iterations.
    The shape of stacked U matrix (Us) is (m, k1*I), where m is the number of features, k1 is the latent dimensions
    and I iterations depending on the convergence status and max iterations.

    Args:
        Us (list of np.ndarray[(1,1), np.floating]) : U matrices from iterations

    Returns:
        (np.ndarray[(1,1), np.floating]) : decomposed U matrix from stacked U matrices with shape (m, k1*I), where m is the number of features, k1 is the latent dimensions
                                           and I iterations depending on the convergence status and max iterations.
    """
    Us = jnp.hstack(Us)
    U, _, _ = svd(Us)
    return U


def create_proxy_matrix(A, U):
    """Calculate proxy matrix P

    Algo5/4
    P is the proxy data matrix with shape (k1*I, n), where n is the number of samples, k1 is the latent dimensions and
    I iterations depending on the convergence status and max iterations.

    Args:
        A (np.ndarray[(1,1), np.floating]) : matrix with shape (m, n), where m and n represent m features and n samples respectively.
        U (np.ndarray[(1,1), np.floating]) : U matrix decomposed from stacked U matrices with shape (m, k1*I), where m is the number of features, k1 is the latent dimensions
                                             and I iterations depending on the convergence status and max iterations.

    Returns:
        (np.ndarray[(1,1), np.floating]) : the proxy data matrix with shape (k1*I, n), where n is the number of samples, k1 is the latent dimensions and I iterations depending on the convergence status and max iterations.
    """
    P = mmdot(U, A)
    return P


def covariance_from_proxy_matrix(P):
    """Calculate covariance matrix from proxy matrix

    Algo5/5
    cov is covariance matrix with shape (k1*I, k1*I).

    Args:
        P (np.ndarray[(1,1), np.floating]) : the proxy data matrix with shape (k1*I, n), where n is the number of samples, k1 is the latent dimensions and I iterations depending on the convergence status and max iterations.

    Returns:
        (np.ndarray[(1,1), np.floating]) : the inner prodcut of proxy data matrix with shape (k1*I, k1*I) as the covariance matrix.
    """
    cov = mmdot(P.T, P.T)
    return cov


def local_V_from_proxy_matrix(P, Vp):
    """Update V matrix

    Use proxy data matrix P (k1*I, n) and eigenvectors Vp (k1*I, k2) from aggregator to get the V matrix with shape (n, k2)

    Args:
        P (np.ndarray[(1,1), np.floating]) : proxy data matrix from edge with shape (k1*I, n)
        Vp (np.ndarray[(1,1), np.floating]) : eigenvectors used for getting the V matrix in edge with shape (k1*I, k2)

    Returns:
        (np.ndarray[(1,1), np.floating]) : the V matrix with shape (n, k2)
    """
    V = mmdot(P, Vp)
    return V


def init_gram_schmidt(M):
    """
    This function supports general usage without SVD process.

    Args:
        M (np.ndarray[(1,1), np.floating]) : The eigenvectors should be placed vertically as M[:,i].

    Returns:
        (np.floating) : the norm of the first partial eigenvector (the rest is distributed on different edges)
        (the list of np.ndarray[(1,), np.floating]) : the list used to store partial eigenvectors in the downstream orthonormalization process
    """
    ortho = [M[:, 0]]

    return jnp.vdot(M[:, 0], M[:, 0]), ortho


def eigenvec_concordance_estimation(GT, SIM, latent_axis=(1, 1), decimal=5):
    if latent_axis[0] != 1:
        GT = GT.T
    if latent_axis[1] != 1:
        SIM = SIM.T

    if GT.shape != SIM.shape:
        raise ValueError(
            f"Inconcordance matrix shapes: {GT.shape} ground truth, {SIM.shape} simulation"
        )

    I = np.identity(GT.shape[1])  # noqa: E741

    def __test(A1, A2):
        A1tA2 = mmdot(A1, A2)
        try:
            np.testing.assert_array_almost_equal(abs(A1tA2), I, decimal=decimal)
        except AssertionError:
            print(
                f"\
                ====================== Inner product ======================\n\
                {A1tA2}\n\
                =========================== A1 ===========================\n\
                {A1}\n\
                =========================== A2 ===========================\n\
                {A2}\n\
                ==========================================================="
            )

    __test(GT, GT)
    __test(SIM, SIM)
    __test(GT, SIM)


class AbsStandardization(abc.ABC):
    def __init__(self):
        pass

    def local_col_nansum(self, A):
        raise NotImplementedError

    def local_imputed_mean(self, A, mean):
        raise NotImplementedError

    def global_mean(self, col_sum, row_count):
        raise NotImplementedError

    def local_ssq(self, A, mean):
        raise NotImplementedError

    def global_var(self, ssq, row_count):
        raise NotImplementedError

    def local_standardize(self, A, var, delete):
        raise NotImplementedError


class FederatedStandardization(AbsStandardization):
    def __init__(self):
        super().__init__()

    def local_col_nansum(self, A):
        col_sum, row_count = stats.nansum(A)
        jump_to = "global_mean"
        return col_sum, row_count, jump_to

    def local_imputed_mean(self, A, mean):
        A = stats.impute_with_mean(A, mean)
        col_sum, row_count = stats.sum_and_count(A)
        jump_to = "global_mean"
        return A, col_sum, row_count, jump_to

    def global_mean(self, col_sum, row_count):
        col_sum = aggregations.SumUp()(*col_sum)
        row_count = aggregations.SumUp()(*row_count)
        mean = col_sum / row_count
        if (col_sum.astype(np.int32) == col_sum).all():
            jump_to = "local_imputed_mean"
        else:
            jump_to = "local_ssq"
        return mean, jump_to

    def local_ssq(self, A, mean):
        A = stats.make_mean_zero(A, mean)
        ssq = stats.sum_of_square(A)
        row_count = A.shape[0]
        return A, ssq, row_count

    def global_var(self, ssq, row_count):
        ssq = aggregations.SumUp()(*ssq)
        row_count = aggregations.SumUp()(*row_count)
        var = ssq / (row_count - 1)
        delete = jnp.where(var == 0)[0]
        var = jnp.delete(var, delete)
        return var, delete

    def local_standardize(self, A, var, delete):
        A = stats.standardize(A, var, delete)
        return A

    @classmethod
    def standalone(cls, As):
        local_sums, local_counts = [], []
        for edge_idx in range(len(As)):
            s, c, _ = cls().local_col_nansum(As[edge_idx])
            local_sums.append(s)
            local_counts.append(c)
        global_mean, _ = cls().global_mean(local_sums, local_counts)

        # global mean from imputed data
        local_sums, local_counts = [], []
        for edge_idx in range(len(As)):
            a, s, c, _ = cls().local_imputed_mean(As[edge_idx], global_mean)
            local_sums.append(s)
            local_counts.append(c)
            As[edge_idx] = a
        global_mean, _ = cls().global_mean(local_sums, local_counts)

        # global variance
        local_ssqs, local_counts = [], []
        for edge_idx in range(len(As)):
            a, ssq, c = cls().local_ssq(As[edge_idx], global_mean)
            local_ssqs.append(ssq)
            local_counts.append(c)
            As[edge_idx] = a
        global_var, delete = cls().global_var(local_ssqs, local_counts)

        # standardize
        std_As = []
        for edge_idx in range(len(As)):
            std_a = cls().local_standardize(As[edge_idx], global_var, delete)
            std_As.append(std_a)

        return std_As


class AbsVerticalSubspaceIteration(abc.ABC):
    def __init__(self):
        pass

    def local_init(self, A, k1):
        raise NotImplementedError

    def global_init(self, n_features, k1):
        raise NotImplementedError

    def update_local_U(self, A, V):
        raise NotImplementedError

    def update_global_U(self, U):
        raise NotImplementedError

    def check_convergence(self, U, prev_U, epsilon):
        raise NotImplementedError

    def update_global_Us(self, U, Us, current_iteration, max_iterations):
        raise NotImplementedError

    def update_local_V(self, A, U, converged, current_iteration, max_iterations):
        raise NotImplementedError


class FederatedVerticalSubspaceIteration(AbsVerticalSubspaceIteration):
    def __init__(self):
        super().__init__()

    def local_init(self, A, k1):
        A = A.T
        V = randn(A.shape[1], k1)
        V, _ = jsp.linalg.qr(V, mode="economic")
        n_features = A.shape[0]
        return A, V, n_features

    def global_init(self, n_features, k1):
        prev_U = init_rand_U(n_features, k1)
        current_iteration = 1
        converged = False
        Us = []
        return prev_U, current_iteration, converged, Us

    def update_local_U(self, A, V):
        U = update_local_U(A, V)
        return U

    def update_global_U(self, U):
        U = aggregations.SumUp()(*U)
        U, _ = orthonormalize(U)
        return U

    def check_convergence(self, U, prev_U, epsilon):
        converged, _ = check_eigenvector_convergence(U, prev_U, epsilon)
        return converged

    def update_global_Us(self, U, Us, current_iteration):
        prev_U, Us, current_iteration = update_Us(U, Us, current_iteration)
        return prev_U, Us, current_iteration

    def update_local_V(self, A, U, converged, current_iteration, max_iterations):
        V = update_local_V(A, U)
        if not converged and current_iteration < max_iterations:
            jump_to = "update_local_U"
        else:
            jump_to = "next"
        return V, jump_to

    @classmethod
    def standalone(cls, As, k1=20, epsilon=1e-9, max_iterations=20):
        # edge init
        local_Gs = []
        for edge_idx in range(len(As)):
            a, g, f = cls().local_init(As[edge_idx], k1)
            As[edge_idx] = a
            local_Gs.append(g)

        # aggregator init
        prev_H, current_iteration, H_converged, Hs = cls().global_init(f, k1)

        # Vertical subspace iterations
        while not H_converged and current_iteration < max_iterations:
            # Update global H
            hs = []
            for edge_idx in range(len(As)):
                h = cls().update_local_U(As[edge_idx], local_Gs[edge_idx])
                hs.append(h)
            global_H = cls().update_global_U(hs)

            # Check convergence & save global H
            H_converged = cls().check_convergence(global_H, prev_H, epsilon)
            prev_H, Hs, current_iteration = cls().update_global_Us(
                global_H, Hs, current_iteration
            )

            # Update local G
            for edge_idx in range(len(As)):
                g, _ = cls().update_local_V(
                    As[edge_idx],
                    global_H,
                    H_converged,
                    current_iteration,
                    max_iterations,
                )
                local_Gs[edge_idx] = g

        return As, Hs, local_Gs


class AbsRandomizedSVD(AbsVerticalSubspaceIteration):
    def __init__(self):
        super().__init__()

    def decompose_global_Us(self, Us):
        raise NotImplementedError

    def compute_local_covariance(self, A, U):
        raise NotImplementedError

    def decompose_global_covariance(self, PtP, k2):
        raise NotImplementedError

    def recontruct_local_V(self, P, Vp):
        raise NotImplementedError


class FederatedRandomizedSVD(FederatedVerticalSubspaceIteration):
    def __init__(self):
        super().__init__()

    def decompose_global_Us(self, Us):
        U = decompose_U_stack(Us)
        return U

    def compute_local_covariance(self, A, U):
        P = create_proxy_matrix(A, U)
        PPt = covariance_from_proxy_matrix(P)
        return P, PPt

    def decompose_global_covariance(self, PPt, k2):
        PPt = aggregations.SumUp()(*PPt)
        Vp = svd(PPt)[0][:, :k2]
        return Vp

    def recontruct_local_V(self, P, Vp):
        V = local_V_from_proxy_matrix(P, Vp)
        return V

    @classmethod
    def standalone(cls, As, Hs, local_Gs, k2=20):
        # Get the projection matrix
        global_H = cls().decompose_global_Us(Hs)

        # Form proxy data matrices to get proxy covariance matrices
        Ps, covs = [], []
        for edge_idx in range(len(As)):
            p, ppt = cls().compute_local_covariance(As[edge_idx], global_H)
            Ps.append(p)
            covs.append(ppt)
        global_H = cls().decompose_global_covariance(covs, k2)

        # Update G matrix and prepare for the orthonormalization
        hs = []
        for edge_idx in range(len(As)):
            g = cls().recontruct_local_V(Ps[edge_idx], global_H)
            h = cls().update_local_U(As[edge_idx], g)
            local_Gs[edge_idx] = g
            hs.append(h)
        global_H = cls().update_global_U(hs)

        return global_H, local_Gs


class FederatedSVD(
    FederatedStandardization, FederatedRandomizedSVD, FederatedGramSchmidt
):
    def __init__(self):
        FederatedStandardization.__init__()
        FederatedRandomizedSVD.__init__()
        FederatedGramSchmidt.__init__()

    @classmethod
    def standalone(cls, As, k1=20, k2=20, epsilon=1e-9, max_iterations=20):
        # Standardization
        std_As = FederatedStandardization.standalone(As)

        # Vertical subspace iterations
        Ast, Hs, local_Gs = FederatedVerticalSubspaceIteration.standalone(
            std_As, k1, epsilon, max_iterations
        )

        # Randomized SVD
        global_H, local_Gs = FederatedRandomizedSVD.standalone(Ast, Hs, local_Gs, k2)

        # Gram-Schmidt Orthonormalization
        local_Gs = FederatedGramSchmidt.standalone(local_Gs)

        return local_Gs, global_H


@jit
def logistic_predict(X, beta):
    """Logistic regression prediction

    Perform sigmoid(X*beta)

    Args:
        X (np.ndarray[(1, 1), np.floating]): Matrix.
        beta (np.ndarray[(1,), np.floating]): Vector.

    Returns:
        np.ndarray[(1,), np.floating]: Vector.
    """
    pred_y = 1 / (1 + jnp.exp(-mvmul(X, beta)))
    return pred_y


@jit
def logistic_residual(y, pred_y):
    """Residual calculation

    Perform y - predicted_y

    Args:
        y (np.ndarray[(1,), np.floating]): Vector.
        pred_y (np.ndarray[(1, 1), np.floating]): Vector.

    Returns:
        np.ndarray[(1, 1), np.floating]: Vector.
    """
    return y - pred_y


@jit
def logistic_gradient(X, residual):
    """Logistic gradient vector

    Perform X.T * (y - predicted_y)

    Args:
        X (np.ndarray[(1, 1), np.floating]): Matrix.
        residual (np.ndarray[(1, 1), np.floating]): Vector.

    Returns:
        np.ndarray[(1, 1), np.floating]: Vector.
    """
    return mvdot(X, residual)


@jit
def logistic_hessian(X, pred_y):
    """Logistic hessian matrix

    Perform jnp.multiply(X.T, (pred_y * (1 - pred_y)).T) * X

    Args:
        X (np.ndarray[(1, 1), np.floating]): Matrix.
        pred_y (np.ndarray[(1, 1), np.floating]): Vector.

    Returns:
        np.ndarray[(1, 1), np.floating]: Matrix.
    """
    return matmul(jnp.multiply(X.T, (pred_y * (1 - pred_y)).T), X)


@jit
def logistic_loglikelihood(y, pred_y):
    """Logistic log likelihood estimation

    Perform SUM(
        y * log(predicted_y + epsilon) +
        (1 - y) * log(1 - predicted_y + epsilon)
    )

    Args:
        y (np.ndarray[(1,), np.floating]): Vector.
        pred_y (np.ndarray[(1, 1), np.floating]): Vector.

    Returns:
        np.ndarray[(1,), np.floating]: float.
    """
    epsilon = jnp.finfo(float).eps
    return jnp.sum(
        y * jnp.log(pred_y + epsilon) + (1 - y) * jnp.log(1 - pred_y + epsilon)
    )


@jit
def batched_logistic_predict(X, beta):
    """Batched logistic regression prediction

    Perform sigmoid(X*beta)

    Args:
        X (np.ndarray[(1, 1, 1), np.floating]): Batched matrix.
        beta (np.ndarray[(1, 1), np.floating]): Batched vector.

    Returns:
        np.ndarray[(1, 1), np.floating]: Batched vector.
    """
    return vmap(logistic_predict, (0, 0), 0)(X, beta)


@jit
def batched_logistic_residual(y, pred_y):
    """Batched residual calculation

    Perform y - predicted_y

    Args:
        y (np.ndarray[(1, 1), np.floating]): Batched vector.
        pred_y (np.ndarray[(1, 1), np.floating]): Batched vector.

    Returns:
        np.ndarray[(1, 1), np.floating]: Batched vector.
    """
    return vmap(logistic_residual, (0, 0), 0)(y, pred_y)


@jit
def batched_logistic_gradient(X, residual):
    """Batched logistic gradient vector

    Perform X.T * (y - predicted_y)

    Args:
        X (np.ndarray[(1, 1, 1), np.floating]): Batched matrix.
        residual (np.ndarray[(1, 1), np.floating]): Batched vector.

    Returns:
        np.ndarray[(1, 1), np.floating]: Batched vector.
    """
    return vmap(logistic_gradient, (0, 0), 0)(X, residual)


@jit
def batched_logistic_hessian(X, pred_y):
    """Batched logistic hessian matrix

    Perform jnp.multiply(X.T, (pred_y * (1 - pred_y)).T) * X

    Args:
        X (np.ndarray[(1, 1, 1), np.floating]): Batched matrix.
        pred_y (np.ndarray[(1, 1), np.floating]): Batched vector.

    Returns:
        np.ndarray[(1, 1, 1), np.floating]: Batched matrix.
    """
    return vmap(logistic_hessian, (0, 0), 0)(X, pred_y)


@jit
def batched_logistic_loglikelihood(y, pred_y):
    """Batched logistic log likelihood estimation

    Perform SUM(
        y * log(predicted_y + epsilon) +
        (1 - y) * log(1 - predicted_y + epsilon)
    )

    Args:
        y (np.ndarray[(1, 1), np.floating]): Batched vector.
        pred_y (np.ndarray[(1, 1), np.floating]): Batched vector.

    Returns:
        np.ndarray[(1,), np.floating]: Batched vector.
    """
    return vmap(logistic_loglikelihood, (0, 0), 0)(y, pred_y)
