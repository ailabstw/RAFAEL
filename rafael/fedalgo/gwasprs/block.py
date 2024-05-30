from abc import ABC, abstractmethod
from typing import List
import copy

import numpy as np
from scipy.sparse import issparse
import jax


class AbstractBlockDiagonalMatrix(ABC):
    def __init__(self):
        pass

    @property
    def ndim(self):
        return 2

    @property
    @abstractmethod
    def nblocks(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def blockshapes(self):
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, key):
        raise NotImplementedError()


class BlockDiagonalMatrixIterator:
    def __init__(self, bd: AbstractBlockDiagonalMatrix) -> None:
        self.__bd = bd
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < self.__bd.nblocks:
            block = self.__bd[self.index]
            self.index += 1
            return block
        else:
            raise StopIteration


class BlockDiagonalMatrix(AbstractBlockDiagonalMatrix):
    """Block diagonal matrix which stores dense numpy matrices separately."""

    def __init__(self, blocks: List[np.ndarray]) -> None:
        super().__init__()
        checks = [
            isinstance(x, (np.ndarray, np.generic, jax.Array)) or issparse(x)
            for x in blocks
        ]
        assert np.all(checks)
        self.__blocks = copy.deepcopy(blocks)

    @property
    def nblocks(self):
        return len(self.__blocks)

    @property
    def blocks(self):
        return self.__blocks

    @property
    def blockshapes(self):
        return [blk.shape for blk in self.blocks]

    def blockshape(self, i: int):
        return self.blocks[i].shape

    @property
    def shape(self):
        return tuple(map(sum, zip(*self.blockshapes)))

    def append(self, x: np.ndarray):
        if isinstance(x, (np.ndarray, np.generic, jax.Array)) or issparse(x):
            return self.__blocks.append(x)
        elif isinstance(x, BlockDiagonalMatrix):
            return self.__blocks.extend(x.blocks)
        else:
            raise Exception(f"cannot append objects of type {type(x)}")

    @classmethod
    def fromlist(cls, ls, array=np.array):
        return cls([array(x) for x in ls])

    @classmethod
    def fromdense(cls, X, nblocks):
        steps = []
        # Check x sizes equal
        for i in range(X.ndim):
            step = X.shape[i] / nblocks
            assert int(step) == step, f"Only equal-sized matrices are supported; \
                found unequal sizes at the {i}-dimensional axis. \
                Please use BlockDiagonalMatrix.fromindex(X, indices)."
            steps.append(int(step))
        X_blocks = []
        for i in range(nblocks):
            slices = tuple(slice(i * s, (i + 1) * s) for s in steps)
            X_blocks.append(X[slices])
        return cls(X_blocks)

    @classmethod
    def fromindex(cls, X, indices):
        """
        Form a BlockDiagonalMatrix from the provided indices.

        Parameters
        ----------
            X : np.ndarray, jnp.array
                Dense matrix.
            indices : List[List[List[int, int]]]
                The indices should be three-dimensional and represented as (n_blocks, n_dim, 2).
                The last dimension should record the start and end indices to form start:end.
        Returns
        -------
            BlockDiagonalMatrix
        """
        indices = np.array(indices)
        assert (
            indices.ndim == 3
        ), f"indices must be three-dimensional, got {indices.ndim}"
        X_blocks = []
        ndim = indices.shape[1]
        for i in range(indices.shape[0]):
            # Get the slices for each dimension of the block
            slices = tuple(
                slice(indices[i][d][0], indices[i][d][1]) for d in range(ndim)
            )
            X_blocks.append(X[slices])
        return cls(X_blocks)

    def __iter__(self):
        return BlockDiagonalMatrixIterator(self)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.__blocks[key]
        elif isinstance(key, slice):
            return self.fromlist(self.__blocks[key])
        else:
            raise KeyError

    def __add__(self, value):
        if isinstance(value, (np.ndarray, np.generic, jax.Array)):
            return self.toarray() + value
        elif isinstance(value, AbstractBlockDiagonalMatrix):
            assert self.blockshapes == value.blockshapes
            return BlockDiagonalMatrix([x + y for (x, y) in zip(self, value)])
        else:
            raise Exception(f"value type of {type(value)} not supported")

    def __matmul__(self, value):
        if isinstance(value, AbstractBlockDiagonalMatrix):
            return BlockDiagonalMatrix(
                [x @ y for (x, y) in zip(self.blocks, value.blocks)]
            )
        elif isinstance(value, (np.ndarray, np.generic, jax.Array)):
            assert value.ndim == 1, f"value must be 1-dimensional, got {value.ndim}"
            rowidx = np.cumsum([0] + [shape[0] for shape in self.blockshapes])
            colidx = np.cumsum([0] + [shape[1] for shape in self.blockshapes])
            res = np.empty(rowidx[-1])
            for i in range(self.nblocks):
                res.view()[rowidx[i] : rowidx[i + 1]] = (
                    self[i] @ value.view()[colidx[i] : colidx[i + 1]]
                )
            return res

    def toarray(self):
        return block_diag(*self.blocks)

    def tolist(self):
        return [blk.tolist() for blk in self.blocks]

    def diagonal(self):
        return np.concatenate([blk.diagonal() for blk in self.blocks])


def block_diag(*arrs):
    if arrs == ():
        arrs = ([],)

    assert all(
        [a.ndim == arrs[0].ndim for a in arrs]
    ), "All arrays must have the same number of dimensions."
    ndim = arrs[0].ndim

    shapes = np.array([a.shape for a in arrs])
    out_dtype = np.result_type(*[a.dtype for a in arrs])
    out = np.zeros(np.sum(shapes, axis=0), dtype=out_dtype)

    idx = np.zeros(ndim, dtype="int")
    for i, sh in enumerate(shapes):
        slices = tuple(slice(idx[j], idx[j] + sh[j]) for j in range(ndim))
        out[slices] = arrs[i]
        idx += np.array(sh)
    return out
