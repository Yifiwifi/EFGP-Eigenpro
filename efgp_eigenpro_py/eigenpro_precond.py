from dataclasses import dataclass, field

import numpy as np


@dataclass
class EigenProPreconditioner:
    """
    Spectral preconditioner defined by top eigenpairs of A.
    P(v) = v - U * ((1 - mu / theta) * (U^H v)).
    """
    eigvals: np.ndarray
    eigvecs: np.ndarray
    mu: float
    scale: np.ndarray = field(init=False, repr=False)
    eigvecs_h: np.ndarray = field(init=False, repr=False)
    _proj_buf: np.ndarray | None = field(init=False, default=None, repr=False)
    _out_buf: np.ndarray | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.eigvals = np.asarray(self.eigvals, dtype=float)
        self.eigvecs = np.asarray(self.eigvecs)
        if self.eigvecs.ndim != 2:
            raise ValueError("eigvecs must be a 2D array.")
        if self.eigvals.ndim != 1:
            raise ValueError("eigvals must be a 1D array.")
        if self.eigvecs.shape[1] != self.eigvals.size:
            raise ValueError("eigvals and eigvecs are inconsistent.")
        if np.any(self.eigvals <= 0):
            raise ValueError("eigvals must be positive.")
        if not (0.0 < self.mu <= float(np.min(self.eigvals))):
            raise ValueError("mu must satisfy 0 < mu <= min(eigvals).")
        self.scale = 1.0 - (self.mu / self.eigvals)
        self.eigvecs_h = self.eigvecs.conj().T

    def apply(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v)
        proj_shape = (self.eigvals.size,) if v.ndim == 1 else (self.eigvals.size, v.shape[1])
        out_shape = v.shape
        dtype = np.result_type(self.eigvecs.dtype, v.dtype)

        proj = self._proj_buf
        if proj is None or proj.shape != proj_shape or proj.dtype != dtype:
            proj = np.empty(proj_shape, dtype=dtype)
            self._proj_buf = proj
        out = self._out_buf
        if out is None or out.shape != out_shape or out.dtype != dtype:
            out = np.empty(out_shape, dtype=dtype)
            self._out_buf = out

        np.dot(self.eigvecs_h, v, out=proj)
        np.multiply(self.scale.reshape(-1, 1) if proj.ndim == 2 else self.scale, proj, out=proj)
        np.dot(self.eigvecs, proj, out=out)
        np.subtract(v, out, out=out)
        return out


def build_preconditioner(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    mu: float,
) -> EigenProPreconditioner:
    return EigenProPreconditioner(eigvals=eigvals, eigvecs=eigvecs, mu=mu)
