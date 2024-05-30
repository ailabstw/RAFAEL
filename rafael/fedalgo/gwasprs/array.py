from typing import NewType

import numpy as np
from scipy.stats import ortho_group
from scipy.sparse import issparse, hstack, vstack
import jax
from jax import numpy as jnp
from jax import scipy as jsp
from jax import random

from . import linalg, iterator


def colons(n):
    return tuple(slice(None) for _ in range(n))


class ArrayIterator:
    def __init__(self, arr, axis=0) -> None:
        self.arr = arr
        self.axis = axis
        self.iter = iterator.IndexIterator(arr.shape[axis])

    def __iter__(self):
        return self

    def __next__(self):
        if not self.iter.is_end():
            slc = next(self.iter)
            idx = tuple([*colons(self.axis), slc])
            return self.arr[idx]
        else:
            raise StopIteration


def concat(xs, axis=1):
    if isinstance(xs[0], (np.ndarray, np.generic)):
        return np.concatenate(xs, axis=axis)
    elif isinstance(xs[0], jax.Array):
        return jnp.concatenate(xs, axis=axis)
    elif issparse(xs[0]):
        if axis == 0:
            return vstack(xs)
        elif axis == 1:
            return hstack(xs)
        else:
            raise ValueError("Only support axis 0 and 1 for sparse arrays.")
    else:
        raise TypeError("Unsupported array type.")


def impute_with(X, val=0.0):
    if isinstance(X, (np.ndarray, np.generic)):
        return np.nan_to_num(X, copy=True, nan=val, posinf=None, neginf=None)
    elif isinstance(X, jax.Array):
        return jnp.nan_to_num(X, copy=True, nan=val, posinf=None, neginf=None)
    else:
        raise TypeError("Unsupported array type.")


def expand_to_2dim(x, axis=-1):
    if isinstance(x, (np.ndarray, np.generic)) and x.ndim == 1:
        x = np.expand_dims(x, axis)
    elif isinstance(x, jax.Array) and x.ndim == 1:
        x = jnp.expand_dims(x, axis)
    else:
        raise TypeError("Unsupported array type.")
    return x


def simulate_genotype_matrix(
    key, shape=(10, 30), r_mask=0.9, c_mask=0.9, impute=False, standardize=False
):
    # simulate genotype matrix with NAs
    X = random.randint(key=key, minval=0, maxval=3, shape=shape).astype("float32")
    mask_ridx = random.choice(key=key, a=shape[0], shape=(int(X.size * 0.9),))
    mask_cidx = random.choice(key=key, a=shape[1], shape=(int(X.size * 0.9),))
    X = X.at[mask_ridx, mask_cidx].set(jnp.nan)

    if impute:
        col_mean = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X = X.at[inds].set(np.take(col_mean, inds[1]))

    if standardize:
        X = (X - jnp.mean(X, axis=0)) / jnp.nanstd(X, axis=0, ddof=1)
        X = jnp.delete(X, jnp.isnan(X[0]), axis=1)

    return X


def _subspace_iteration(A, G):
    H = linalg.mmdot(A.T, G)
    H, R = jsp.linalg.qr(H, mode="economic")
    G = linalg.mmdot(A, H)
    G, R = jsp.linalg.qr(G, mode="economic")
    return G, R


def simulate_eigenvectors(n, m, k, seed=42, iterations=10):
    A = linalg.randn(n, m, seed).T
    G = ortho_group.rvs(dim=n)
    for _ in range(iterations):
        G, R = _subspace_iteration(A, G)
    return G[:, :k], R[:, :k]
