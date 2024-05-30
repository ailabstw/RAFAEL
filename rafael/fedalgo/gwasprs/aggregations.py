from abc import abstractmethod

import numpy as np
from scipy.sparse import issparse
import jax
import jax.numpy as jnp
from ordered_set import OrderedSet

from . import block


class Aggregation:
    def __call__(self, *xs):
        self.n = len(xs)
        assert self.n > 0, "xs should not be empty"

        x = xs[0]

        if x is None:
            return None
        elif isinstance(x, list) and (
            isinstance(x[0], (np.ndarray, np.generic, jax.Array)) or issparse(x[0])
        ):
            return self.aggregate_list_of_array(*xs)
        elif isinstance(x, list):
            return self.aggregate_list_of_list(*xs)
        elif isinstance(x, int) or isinstance(x, float):
            return self.aggregate_scalars(*xs)
        elif isinstance(x, (np.ndarray, np.generic, jax.Array)) or issparse(x):
            x_types = set(type(a) for a in xs)
            assert (
                len(x_types) == 1
            ), f"All array types must be consistent, but got {x_types}."
            return self.aggregate_arrays(*xs)
        elif isinstance(x, block.BlockDiagonalMatrix):
            return self.aggregate_block_diags(*xs)
        else:
            raise NotImplementedError(
                f"{type(x)} is not supported, expected int, float, np.ndarray, scipy sparse array or list of np.ndarray"
            )

    @abstractmethod
    def aggregate_list_of_array(self, *xs):
        raise NotImplementedError(
            "Abstract aggregation for list of array is not implemented yet"
        )

    @abstractmethod
    def aggregate_list_of_list(self, *xs):
        raise NotImplementedError(
            "Abstract aggregation for list is not implemented yet"
        )

    @abstractmethod
    def aggregate_scalars(self, *xs):
        raise NotImplementedError(
            "Abstract aggregation for scalars is not implemented yet"
        )

    @abstractmethod
    def aggregate_arrays(self, *xs):
        raise NotImplementedError(
            "Abstract aggregation for arrays is not implemented yet"
        )

    @abstractmethod
    def aggregate_block_diags(self, *xs):
        raise NotImplementedError(
            "Abstract aggregation for `BlockDiagonalMatrix` is not implemented yet"
        )


class SumUp(Aggregation):
    def aggregate_list_of_array(self, *xs):
        agg_weight = []
        len_of_weight = len(xs[0])
        for j in range(len_of_weight):
            tmp_array = xs[0][j]
            for i in range(1, self.n):
                tmp_array = tmp_array + xs[i][j]

            agg_weight.append(tmp_array)

        return agg_weight

    def aggregate_block_diags(self, *xs):
        result = xs[0]
        for i in range(1, self.n):
            result = result + xs[i]

        return result

    def aggregate_scalars(self, *xs):
        return sum(xs)

    def aggregate_arrays(self, *xs):
        result = xs[0]
        for i in range(1, self.n):
            result = result + xs[i]

        return result


class Intersect(Aggregation):
    def aggregate_list_of_array(self, *xs):
        raise NotImplementedError("InterSect for list of array is not implemented yet")

    def aggregate_scalars(self, *xs):
        raise NotImplementedError("InterSect for scalars is not implemented yet")

    def aggregate_arrays(self, *xs):
        if len(xs) == 1:
            return xs[0]

        intersected = OrderedSet(xs[0].tolist())
        for x in xs[1:]:
            intersected.intersection_update(x.tolist())

        if isinstance(xs[0], (np.ndarray, np.generic)):
            return np.array(list(intersected))
        elif isinstance(xs[0], jax.Array):
            return jnp.array(list(intersected))
        elif issparse(xs[0]):
            raise NotImplementedError
        else:
            raise TypeError("Unsupported array type.")

    def aggregate_list_of_list(self, *xs):
        if len(xs) == 1:
            return xs[0]

        intersected = OrderedSet(xs[0])
        for x in xs[1:]:
            intersected.intersection_update(x)

        return list(intersected)


class IsSame(Aggregation):
    def aggregate_list_of_array(self, xs):
        raise NotImplementedError("CheckSame for list of array is not implemented yet")

    def aggregate_scalars(self, xs):
        if len(set(xs)) > 1:
            raise RuntimeError(f"weight from client are not the same {xs}")

        return xs[0]

    def aggregate_arrays(self, xs):
        if not all_equal(xs):
            raise RuntimeError(f"weight from client are not the same {xs}")

        return xs[0]


def all_equal(iterator):
    try:
        iterator = iter(iterator)
        first = next(iterator)
        return all(np.array_equal(first, rest) for rest in iterator)
    except StopIteration:
        return True
