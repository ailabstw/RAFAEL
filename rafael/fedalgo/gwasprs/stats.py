import numpy as np
import scipy.stats as stats
import jax
from jax import jit, vmap, pmap
from jax import numpy as jnp
from jax.typing import ArrayLike

from . import linalg, utils


def make_mean_zero(A, mean):
    # Make the SNP mean = 0
    return A - mean


def sum_of_square(A):
    """Sum of square

    Make column-wise mean of A equal to 0 and calculate the column-wise sum of squares.
    original genotype_scaling_var_step

    Args:
        A (np.ndarray[(1, 1), np.floating]) : Genotype matrix with shape (samples, SNPs)
        mean (np.ndarray[(1,), np.floating]) : Vector

    Returns:
        (np.ndarray[(1, 1), np.floating]) : modified A matrix
        (np.ndarray[(1,), np.floating]) : sum of square vector
    """
    return jnp.sum(jnp.square(A), axis=0)


@jit
def normalize(norms, ortho):
    """Normalize the length of eigenvector

    Make the length of eigenvector equal to 1.
    original normalize_step

    Args:
        norms (np.ndarray[(1,), np.floating]) : Vector
        ortho (list of np.ndarray[(1,), np.floating]) : List

    Returns:
        (np.ndarray[(1, 1), np.floating]) : normalized eigenvectors as a matrix
    """
    ortho = jnp.asarray(ortho)
    norms = 1 / jnp.sqrt(jnp.expand_dims(jnp.asarray(norms), -1))
    return (norms * ortho).T


def unnorm_autocovariance(
    X: "np.ndarray[(1, 1), np.floating]",
) -> "np.ndarray[(1, 1), np.floating]":
    return linalg.mmdot(X, X)


def unnorm_covariance(
    X: "np.ndarray[(1, 1), np.floating]", y: "np.ndarray[(1,), np.floating]"
):
    return linalg.mvdot(X, y)


def batched_unnorm_autocovariance(
    X: "np.ndarray[(1, 1, 1), np.floating]", acceleration="single"
) -> "np.ndarray[(1, 1, 1), np.floating]":
    if acceleration == "single":
        return linalg.batched_mmdot(X, X)
    elif acceleration == "pmap":
        pmap_func = pmap(linalg.batched_mmdot, in_axes=0, out_axes=0)
        ncores = utils.jax_dev_count()
        batch, nsample, ndims = X.shape
        minibatch, remainder = divmod(batch, ncores)
        A = np.reshape(
            X[: (minibatch * ncores), :, :], (ncores, minibatch, nsample, ndims)
        )
        Y = np.reshape(pmap_func(A, A), (-1, ndims, ndims))

        if remainder != 0:
            B = X[(minibatch * ncores) :, :, :]
            Z = linalg.batched_mmdot(B, B)
            Y = np.concatenate((Y, Z), axis=0)
        return Y
    else:
        raise ValueError(f"{acceleration} acceleration is not supported.")


def batched_unnorm_covariance(
    X: "np.ndarray[(1, 1, 1), np.floating]", y: "np.ndarray[(1, 1), np.floating]"
):
    return linalg.batched_mvdot(X, y)


def blocked_unnorm_autocovariance(
    X: "np.ndarray[(1, 1), np.floating]",
) -> "np.ndarray[(1, 1), np.floating]":
    return linalg.mmdot(X, X)


def blocked_unnorm_covariance(
    X: "np.ndarray[(1, 1), np.floating]", y: "np.ndarray[(1,), np.floating]"
):
    return linalg.mvdot(X, y)


def t_dist_pvalue(t_stat, df):
    return 2.0 * stats.t.sf(np.abs(t_stat), df)


# Logistic


@jit
def _cal_square_tstat(beta, inv_hessian):
    std = jnp.sqrt(inv_hessian.diagonal())
    t_stat = beta / std
    square_t_stat = jnp.square(t_stat)
    return t_stat, square_t_stat


def chi2_dist_pvalue(square_t_stat, df=1):
    p_value = stats.chi2.sf(np.array(square_t_stat), df)
    return p_value


def logistic_stats(beta, inv_hessian):
    """
    beta: [feat]
    inv_hessian: [feat, feat]
    """
    t_stat, square_t_stat = _cal_square_tstat(beta, inv_hessian)
    p_value = chi2_dist_pvalue(square_t_stat)
    return t_stat, p_value


def batched_logistic_stats(beta, inv_hessian):
    """
    beta: [batch, feat]
    inv_hessian: [batch, feat, feat]
    """
    t_stat, square_t_stat = vmap(_cal_square_tstat, (0, 0), (0, 0))(beta, inv_hessian)
    p_value = chi2_dist_pvalue(square_t_stat)
    return t_stat, p_value


# PCA


def _vectorize(func, in_axes, out_axes):
    return vmap(jit(func), in_axes=in_axes, out_axes=out_axes)


def nansum(A):
    """Sum of matrix discarding NAs

    Perform nansum.
    original genotype_impute_step

    Args:
        A (np.ndarray[(1, 1), np.floating]) : Genotype matrix with shape (samples, SNPs)

    Returns
        (np.ndarray[(1,), np.floating]) : nansum of SNP as a vector
        (int) : sample count
    """
    snp_sum, sample_count = _vectorize(linalg.nansum, 1, 0)(A)
    return snp_sum, sample_count


def impute_with_mean(A: ArrayLike, mean: ArrayLike):
    """
    Fill NAs with column means.
    Deprecate the jax implementation due to the resource consumption
    and the lack of `jit` advantage.
    The following way is really fast.
    """
    if isinstance(A, jax.Array):
        isjax = True
        A = np.array(A)
        mean = np.array(mean)
        
    na_indices = np.where(np.isnan(A))
    A[na_indices] = np.take(mean, na_indices[1])

    if isjax:
        A = jnp.array(A)
        mean = jnp.array(mean)
    
    return A


def sum_and_count(A):
    col_sum = _vectorize(jnp.sum, 1, 0)(A)
    count = A.shape[0]
    return col_sum, count


def standardize(A, global_var, deleted):
    """Standardize the preprocessed genotype matrix

    Perform the final step of standardization
    original genotype_standardization_step

    Args:
        A (np.ndarray[(1, 1), np.floating]) : Imputed genotype matrix with SNP mean=0
        global_var (np.ndarray[(1,), np.floating]) : global variance of SNP
        deleted (np.ndarray[(1,), np.floating]) : boolean vector for whose variance = 0

    Returns:
        np.ndarray[(1, 1), np.floating]) : Standardized genotype matrix
    """
    A = jnp.delete(A, deleted, axis=1)
    A = A / jnp.sqrt(global_var)

    return A
