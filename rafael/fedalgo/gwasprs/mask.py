import numpy as np
import jax
from jax import numpy as jnp


def isnonnan(X: np.ndarray, axis=1):
    if isinstance(X, (np.ndarray, np.generic)):
        return np.sum(np.isnan(X), axis=axis) == 0
    elif isinstance(X, jax.Array):
        return jnp.sum(jnp.isnan(X), axis=axis) == 0
    else:
        raise TypeError("X must be either a numpy array or a jax array.")


def dropnan(X: np.ndarray, axis=1):
    nonnan_idx = np.sum(np.isnan(X), axis=axis) == 0
    return X[nonnan_idx]


def get_mask(X: "np.ndarray[(1, 1), np.floating]"):
    if isinstance(X, (np.ndarray, np.generic)):
        return np.expand_dims(isnonnan(X, axis=1), -1)
    elif isinstance(X, jax.Array):
        return jnp.expand_dims(isnonnan(X, axis=1), -1)
    else:
        raise TypeError("X must be either a numpy array or a jax array.")


def nonnan_count(x, axis=0):
    """
    Number of non-NaN samples
    """
    if isinstance(x, (np.ndarray, np.generic)):
        return np.sum(np.logical_not(np.isnan(x)), axis=axis)
    elif isinstance(x, jax.Array):
        return jnp.sum(jnp.logical_not(jnp.isnan(x)), axis=axis)
    else:
        raise TypeError("X must be either a numpy array or a jax array.")
