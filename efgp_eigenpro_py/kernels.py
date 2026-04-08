from dataclasses import dataclass
from typing import Callable

import math
import numpy as np


@dataclass
class KernelSpec:
    """
    Translation-invariant kernel specification.

    k should accept array of radial distances and return spatial kernel values.
    khat should accept array of radial frequencies and return Fourier spectrum.
    """
    fam: str
    dim: int
    lengthscale: float
    variance: float
    nu: float | None
    k: Callable[[np.ndarray], np.ndarray]
    khat: Callable[[np.ndarray], np.ndarray]

    def to_dict(self) -> dict:
        return {
            "fam": self.fam,
            "dim": self.dim,
            "lengthscale": self.lengthscale,
            "variance": self.variance,
            "nu": self.nu,
        }

    def __repr__(self) -> str:
        return (
            "KernelSpec("
            f"fam={self.fam!r}, "
            f"dim={self.dim}, "
            f"lengthscale={self.lengthscale}, "
            f"variance={self.variance}, "
            f"nu={self.nu})"
        )


def make_squared_exponential(lengthscale: float, dim: int, variance: float = 1.0) -> KernelSpec:
    """
    Squared-exponential kernel spectrum with 2*pi convention.
    """
    if dim < 1:
        raise ValueError("dim must be at least 1.")
    if lengthscale <= 0:
        raise ValueError("lengthscale must be positive.")
    if variance <= 0:
        raise ValueError("variance must be positive.")
    
    # variance = 1 as default
    # SE kernel (radial): k(r) = variance * exp(-0.5 * r^2 / l^2)
    # Fourier spectrum (2*pi convention):
    # khat(|xi|) = variance * (2*pi*l^2)^(d/2) * exp(-2*pi^2*l^2*|xi|^2)
    def k(r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        return variance * np.exp(-0.5 * (r**2) / (lengthscale**2))

    def khat(r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        scale = (2.0 * np.pi * lengthscale**2) ** (dim / 2)
        return variance * scale * np.exp(-(2.0 * np.pi**2) * lengthscale**2 * r**2)

    return KernelSpec(
        fam="squared-exponential",
        dim=dim,
        lengthscale=lengthscale,
        variance=variance,
        nu=None,
        k=k,
        khat=khat,
    )


def make_matern(lengthscale: float, nu: float, dim: int, variance: float = 1.0) -> KernelSpec:
    """
    Matern kernel spectrum with 2*pi convention.
    """
    if dim < 1:
        raise ValueError("dim must be at least 1.")
    if lengthscale <= 0:
        raise ValueError("lengthscale must be positive.")
    if variance <= 0:
        raise ValueError("variance must be positive.")
    if nu <= 0:
        raise ValueError("nu must be positive.")

    # Matern kernel (radial):
    # variance = 1 as default
    # k(r) = variance * (2^(1-nu)/Gamma(nu)) * (sqrt(2*nu)*r/l)^nu
    #        * K_nu(sqrt(2*nu)*r/l)
    # Fourier spectrum (2*pi convention):
    # khat(|xi|) = variance * scaling * (2*nu/l^2 + 4*pi^2*|xi|^2)^(-(nu + d/2))
    scaling = (
        (2.0 * math.sqrt(math.pi)) ** dim
        * math.gamma(nu + dim / 2)
        * (2.0 * nu) ** nu
        / (math.gamma(nu) * lengthscale ** (2.0 * nu))
    )

    def k(r: np.ndarray) -> np.ndarray:
        r = np.abs(np.asarray(r, dtype=float))
        try:
            from scipy import special as sp  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime dependency check
            raise RuntimeError("scipy is required for Matern spatial kernel.") from exc
        scaled = math.sqrt(2.0 * nu) * r / lengthscale
        prefactor = variance * (2.0 ** (1.0 - nu)) / math.gamma(nu)
        out = np.empty_like(scaled, dtype=float)
        zero_mask = scaled == 0
        out[zero_mask] = variance
        if np.any(~zero_mask):
            scaled_nz = scaled[~zero_mask]
            out[~zero_mask] = prefactor * np.power(scaled_nz, nu) * sp.kv(nu, scaled_nz)
        return out

    def khat(r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        base = 2.0 * nu / lengthscale**2 + (4.0 * np.pi**2) * r**2
        return variance * scaling * np.power(base, -(nu + dim / 2))

    return KernelSpec(
        fam="matern",
        dim=dim,
        lengthscale=lengthscale,
        variance=variance,
        nu=nu,
        k=k,
        khat=khat,
    )
