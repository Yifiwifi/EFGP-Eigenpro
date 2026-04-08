from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .kernels import KernelSpec


@dataclass
class GridSpec:
    xis: np.ndarray
    h: float
    mtot: int
    hm: int


def choose_grid_params(
    kernel: KernelSpec, # dim inside
    eps: float,
    L: float,
    use_integral: bool = False,
    l2scaled: bool = False,
    # choose according to Barnett2023 uniform error formula
) -> GridSpec:
    """
    Ported from gp-shootout get_xis with minimal changes.
    TODO: replace with Barnett2023 uniform error formula if needed.
    """
    fam = kernel.fam.lower()
    dim = kernel.dim
    eps_use = eps

    if use_integral:
        raise NotImplementedError("Integral tail estimate not ported yet.")

    if fam == "matern":
        if kernel.nu is None:
            raise ValueError("Matern kernel requires nu.")
        l = kernel.lengthscale
        nu = kernel.nu
        eps_use = eps / kernel.variance
        if l2scaled:
            # L2 scaling term from EFGP heuristics.
            rl2sq = (
                (2 * nu / np.pi / l**2) ** (dim / 2)
                * kernel.khat(0.0) ** 2 / 2
                * np.math.gamma(dim / 2 + 2 * nu)
                / np.math.gamma(dim + 2 * nu)
                * 2 ** (-dim / 2)
            )
            eps_use = eps * np.sqrt(rl2sq)

        eps = eps_use
        h = 1.0 / (L + 0.85 * l / np.sqrt(nu) * np.log(1.0 / eps))
        hm = np.ceil(
            (np.pi ** (nu + dim / 2) * l ** (2 * nu) * eps / 0.15)
            ** (-1.0 / (2 * nu + dim / 2))
            / h
        ).astype(int)

    elif fam == "squared-exponential":
        l = kernel.lengthscale
        eps_use = eps / kernel.variance
        if l2scaled:
            rl2sq = kernel.variance ** 2 * (np.sqrt(np.pi) * l**2) ** dim
            eps_use = eps * np.sqrt(rl2sq)
        eps = eps_use
        h = 1.0 / (1 + l * np.sqrt(2 * np.log(4 * dim * 3**dim / eps)))
        hm = np.ceil(
            np.sqrt(np.log(dim * (4 ** (dim + 1) * dim) / eps) / 2) / np.pi / l / h
        ).astype(int)
    else:
        raise NotImplementedError(f"Unknown kernel family: {kernel.fam}")

    xis = np.arange(-hm, hm + 1) * h
    mtot = xis.size
    return GridSpec(xis=xis, h=h, mtot=mtot, hm=hm)


def generate_multi_index(m: int, dim: int) -> np.ndarray:
    """
    Returns array of shape (M, dim) for J_m = {-m,...,m}^dim.
    """
    grid = [np.arange(-m, m + 1)] * dim
    mesh = np.meshgrid(*grid, indexing="ij")
    return np.stack([g.reshape(-1) for g in mesh], axis=1)


def radial_grid(xis: np.ndarray, dim: int) -> np.ndarray:
    """
    Returns radial grid for isotropic kernels.
    """
    if dim == 1:
        return np.abs(xis)
    grid = np.meshgrid(*([xis] * dim), indexing="ij")
    rsq = np.zeros_like(grid[0], dtype=float)
    for g in grid:
        rsq = rsq + g**2
    return np.sqrt(rsq)


def basis_weights(kernel: KernelSpec, xis: np.ndarray, h: float) -> np.ndarray:
    """
    Computes sqrt(h^d * khat(|xi|)) on the Cartesian grid.
    We do not compute Phi or A here, but only the basis weights, or d of D=diag(d) in EFGP(17).
    Compute according to EFGP (14). The result is flattened in C order to match
    meshgrid(indexing="ij") and generate_multi_index ordering.
    """

    rs = radial_grid(xis, kernel.dim)
    weights = np.sqrt(kernel.khat(rs) * (h ** kernel.dim))
    return weights.reshape(-1)
