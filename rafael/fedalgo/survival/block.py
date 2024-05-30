import numpy as np

class BlockMatrix:
    def __init__(self, Ms, axis=1):
        """
        Parameters
        ----------
            Ms : list of lists of arrays
                The list stores the lists of arrays
            axis : int
                The axis of unequal sizes.
                For example, the matrix shapes in a Ms[0]:
                [(3,10), (3,2), (3,4)], the axis=1,
                [(10,3), (2,3), (4,2)], the axis=0.
                This will affect the indexing output.
        """
        self.blocks = Ms
        self.__nc = len(self.blocks)
        self.__nd = len(self.blocks[0])
        # Each matrix in the 2nd layer list has different shape in axis 1, the axis=1,
        # similarily, each matrix has different shape in axis 0, the axis=0.
        self.axis = axis

    @property
    def shape(self):
        return (self.__nc, self.__nd)

    @property
    def csizes(self):
        """
        Returns
        -------
            csizes : list of int
                The number of samples for each block.
        """
        return list(map(lambda mc: mc[0].shape[0], self.blocks))

    @property
    def dsizes(self):
        """
        Returns
        -------
            dsizes : list of int
                The number of features for each block.
        """
        return list(map(lambda m: m.shape[1], self.blocks[0]))

    @staticmethod
    def subset(ls, idx):
        if isinstance(idx, slice):
            return ls[idx]
        elif isinstance(idx, int):
            return [ls[idx]]
        elif isinstance(idx, (list, tuple)):
            return list(map(lambda i: ls[i], idx))

    def __getitem__(self, indices):
        """
        Parameters
        ----------
            indices : slice, int or list
                The index to indicate the (cth, dth) block.

        Returns
        -------
            block : np.array
                The concatenated matrix.

        Examples
        --------
            # The example of axis=1
            >>> x11 = np.array([1]*4).reshape(2,2)
            >>> x12 = np.array([2]*6).reshape(2,3)
            >>> x21 = np.array([3]*6).reshape(3,2)
            >>> x22 = np.array([4]*9).reshape(3,3)
            >>> np.block([[x11,x12],[x21,x22]])
            array([[1, 1, 2, 2, 2],
                   [1, 1, 2, 2, 2],
                   [3, 3, 4, 4, 4],
                   [3, 3, 4, 4, 4],
                   [3, 3, 4, 4, 4]])
            >>> blk = BlockMatrix([[x11,x12],[x21,x22]])
            >>> blk[0, :]
            array([[1, 1, 2, 2, 2],
                   [1, 1, 2, 2, 2]])
            >>> blk[:, 0]
            array([[1, 1],
                   [1, 1],
                   [3, 3],
                   [3, 3],
                   [3, 3]])

            # The sample of axis=0
            >>> x11 = np.array([1]*4).reshape(2,2)
            >>> x12 = np.array([2]*6).reshape(3,2)
            >>> x21 = np.array([3]*8).reshape(2,4)
            >>> x22 = np.array([4]*12).reshape(3,4)
            >>> blk = BlockMatrix([[x11,x12],[x21,x22]])
            >>> blk[0, :]
            array([[1, 1],
                   [1, 1],
                   [2, 2],
                   [2, 2],
                   [2, 2]])
            >>> blk[:, 0]
            array([[1, 1, 3, 3, 3, 3],
                   [1, 1, 3, 3, 3, 3]])
        """
        M = list(
            map(
                lambda m: self.subset(m, indices[1]),
                self.subset(self.blocks, indices[0]),
            )
        )
        if self.axis == 0:
            return np.concatenate([np.concatenate(mc) for mc in M], axis=1)
        elif self.axis == 1:
            return np.block(M)