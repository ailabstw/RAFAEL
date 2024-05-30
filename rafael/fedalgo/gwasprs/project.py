import abc

from jax import numpy as jnp

from . import stats, linalg, aggregations


def compute_residuals(M, orthogonalized, eigen_idx, norms):
    """Calculate the local residuals

    When orthogonalizing ith eigenvector, there are i-1 values in the residuals.
    (u.v)/(u norm), where u is the orthogonalized vector and v is the eigenvector to be orthogonalized.

    Args:
        G (np.ndarray[(1,1), np.floating]) : The G matrix with shape (n, k2), where n and k2 represent the number of samples and the latent dimensions decided in gwasprs.linalg.decompose_cov_matrices step.
        Ortho (list of np.ndarray[(1,), np.floating]) : the list stores i-1 orthogonalized vectors (n,) of G matrix.
        eigen_idx (int) : the index represents ith eigenvector, e.g. 2nd eigenvector: eigen_idx=1.
        norms (list of np.floating) : list stored i-1 real global norms

    Returns:
        (list of np.floating) : i-1 residuals used for orthogonalized ith eigenvector.
    """
    residuals = []
    for res_idx in range(eigen_idx):
        u = orthogonalized[res_idx]
        v = M[:, eigen_idx]
        r = jnp.vdot(u, v) / norms[res_idx]
        residuals.append(r)

    return residuals


def update_ortho_vectors(ortho_v, orthogonalized):
    orthogonalized.append(ortho_v)


def compute_norm(ortho_v):
    return jnp.vdot(ortho_v, ortho_v)


class AbsGramSchmidt(abc.ABC):
    def __init__(self):
        pass

    def local_first_norm(self, M):
        raise NotImplementedError

    def global_first_norm(self, partial_norm):
        raise NotImplementedError

    def local_residuals(self, M, ortho_v, eigen_idx, norms):
        raise NotImplementedError

    def global_residuals(self, residuals):
        raise NotImplementedError

    def local_nth_norm(self, M, ortho_v, eigen_idx, residuals):
        raise NotImplementedError

    def global_nth_norm(self, global_norms, partial_norm, eigen_idx, k2):
        raise NotImplementedError

    def local_normalization(self, global_norms, ortho_v):
        raise NotImplementedError


class FederatedGramSchmidt(AbsGramSchmidt):
    def __init__(self):
        super().__init__()

    def local_first_norm(self, M):
        partial_norm, orthogonalized = linalg.init_gram_schmidt(M)
        return partial_norm, orthogonalized

    def global_first_norm(self, partial_norm):
        global_norms = [aggregations.SumUp()(*partial_norm)]
        eigen_idx = 1
        return global_norms, eigen_idx

    def local_residuals(self, M, orthogonalized, eigen_idx, norms):
        residuals = compute_residuals(M, orthogonalized, eigen_idx, norms)
        return residuals

    def global_residuals(self, residuals):
        residuals = aggregations.SumUp()(*residuals)
        return residuals

    def local_nth_norm(self, M, orthogonalized, eigen_idx, residuals):
        ortho_v = linalg.orthogonalize(M[:, eigen_idx], orthogonalized, residuals)
        update_ortho_vectors(ortho_v, orthogonalized)
        partial_norm = compute_norm(ortho_v)
        return partial_norm

    def global_nth_norm(self, global_norms, partial_norm, eigen_idx, k2):
        norm = aggregations.SumUp()(*partial_norm)
        global_norms.append(norm)
        eigen_idx += 1
        if eigen_idx < k2 - 1:
            jump_to = "local_residuals"
        else:
            jump_to = "next"
        return global_norms, eigen_idx, jump_to

    def local_normalization(self, global_norms, orthogonalized):
        M = stats.normalize(global_norms, orthogonalized)
        return M

    @classmethod
    def standalone(cls, MTXs):
        # First eigenvector
        partial_norms, orthos = [], []
        for edge_idx in range(len(MTXs)):
            norm, ortho = cls().local_first_norm(MTXs[edge_idx])
            partial_norms.append(norm)
            orthos.append(ortho)
        global_norms, _ = cls().global_first_norm(partial_norms)

        # Rest
        for eigen_idx in range(1, MTXs[0].shape[1]):
            # Calculate residuals
            residuals = []
            for edge_idx in range(len(MTXs)):
                res = cls().local_residuals(
                    MTXs[edge_idx], orthos[edge_idx], eigen_idx, global_norms
                )
                residuals.append(res)
            residuals = cls().global_residuals(residuals)

            # Calculate norms
            partial_norms = []
            for edge_idx in range(len(MTXs)):
                norm = cls().local_nth_norm(
                    MTXs[edge_idx], orthos[edge_idx], eigen_idx, residuals
                )
                partial_norms.append(norm)
            global_norms, _, _ = cls().global_nth_norm(
                global_norms, partial_norms, eigen_idx, MTXs[0].shape[1]
            )

        # Normalize the length to 1
        orthonormal = []
        for edge_idx in range(len(MTXs)):
            mtx = cls().local_normalization(global_norms, orthos[edge_idx])
            orthonormal.append(mtx)

        return orthonormal
