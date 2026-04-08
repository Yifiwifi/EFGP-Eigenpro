from dataclasses import dataclass
from typing import Optional


@dataclass
class KernelConfig:
    fam: str
    dim: int
    lengthscale: float
    variance: float = 1.0
    nu: Optional[float] = None


@dataclass
class GridConfig:
    eps: float
    use_integral: bool = False
    l2scaled: bool = False


@dataclass
class SolverConfig:
    reg_lambda: float
    cg_tol: float
    cg_maxiter: int
    nufft_tol: float
    top_q: int
    stop_relres: float
