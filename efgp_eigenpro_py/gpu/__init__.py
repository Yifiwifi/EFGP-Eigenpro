"""
GPU scaffolding for EFGP/EigenPro migration.

Staged migration layout:
- V1: GPU matvec + GPU CG; NUFFT may still use CPU ``finufft`` then transfer (see ``v1_ops``)
- V2: preconditioner apply P(v) on GPU (eigenspace still CPU)
- V3: eigenspace estimation on GPU
"""

from .backends import BackendConfig, GPUBackendBundle, build_gpu_backend_bundle
from .contexts import GPUDataContext, GPUOperatorContext, ensure_gpu_data_context
from .versions import (
    run_v1_pure_efgp,
    run_v2_with_preconditioner_apply,
    run_v3_full_gpu_eigenspace,
    run_v4_dominant_subspace_preconditioner,
)
from .v2_preconditioner import (
    GPUDominantSubspacePreconditionerData,
    GPUPreconditionerData,
    apply_preconditioner_dominant_subspace,
    apply_preconditioner_v2,
    build_dominant_subspace_preconditioner,
    build_gpu_dominant_subspace_data,
    build_gpu_preconditioner_data,
)
from .iterative_solvers import cg_solve_gpu, pcg_solve_gpu
from .v1_ops import (
    apply_A_v1,
    gpu_precompute_v1,
    predict_v1,
    solve_beta_plain_cg_v1,
)

__all__ = [
    "BackendConfig",
    "GPUBackendBundle",
    "build_gpu_backend_bundle",
    "GPUDataContext",
    "GPUOperatorContext",
    "ensure_gpu_data_context",
    "run_v1_pure_efgp",
    "run_v2_with_preconditioner_apply",
    "run_v3_full_gpu_eigenspace",
    "run_v4_dominant_subspace_preconditioner",
    "GPUDominantSubspacePreconditionerData",
    "GPUPreconditionerData",
    "build_gpu_dominant_subspace_data",
    "build_dominant_subspace_preconditioner",
    "apply_preconditioner_dominant_subspace",
    "build_gpu_preconditioner_data",
    "apply_preconditioner_v2",
    "gpu_precompute_v1",
    "apply_A_v1",
    "predict_v1",
    "solve_beta_plain_cg_v1",
    "cg_solve_gpu",
    "pcg_solve_gpu",
]
