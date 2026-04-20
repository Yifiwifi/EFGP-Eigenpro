from __future__ import annotations

import math
import time
from typing import Any, Callable, Optional


def _sync_device(xp: Any) -> None:
    cuda = getattr(xp, "cuda", None)
    if cuda is not None:
        cuda.Stream.null.synchronize()


def _ensure_workspace_vector(op_ctx: Any, xp: Any, name: str, size: int, dtype: Any) -> Any:
    buf = getattr(op_ctx, name, None)
    if buf is None or getattr(buf, "size", 0) != int(size) or buf.dtype != dtype:
        buf = xp.empty((int(size),), dtype=dtype)
        setattr(op_ctx, name, buf)
    return buf


def cg_solve_gpu(
    backend: Any,
    matvec: Callable[[Any, Any], None],
    b: Any,
    op_ctx: Any,
    tol: float,
    maxiter: int,
    *,
    return_stats: bool = False,
    work_prefix: str = "cg",
) -> tuple[Any, int, float] | tuple[Any, int, float, dict[str, float]]:
    xp = backend.xp
    dtype = xp.complex128
    b = xp.asarray(b, dtype=dtype).reshape(-1)
    n = int(b.size)

    x = _ensure_workspace_vector(op_ctx, xp, f"{work_prefix}_x", n, dtype)
    r = _ensure_workspace_vector(op_ctx, xp, f"{work_prefix}_r", n, dtype)
    p = _ensure_workspace_vector(op_ctx, xp, f"{work_prefix}_p", n, dtype)
    Ap = _ensure_workspace_vector(op_ctx, xp, f"{work_prefix}_ap", n, dtype)

    t_matvec_total = 0.0
    n_matvec = 0

    def _matvec_in(v: Any, out: Any) -> None:
        nonlocal t_matvec_total, n_matvec
        _sync_device(xp)
        t0 = time.perf_counter()
        matvec(v, out)
        _sync_device(xp)
        t_matvec_total += time.perf_counter() - t0
        n_matvec += 1

    x.fill(0)
    _matvec_in(x, Ap)
    xp.subtract(b, Ap, out=r)
    xp.copyto(p, r)

    rsold = float(xp.real(backend.linalg.vdot(r, r)))
    norm_b = max(float(backend.linalg.norm(b)), 1e-30)
    it = 0

    if maxiter <= 0:
        relres = float(backend.linalg.norm(r) / norm_b)
        if not return_stats:
            return x, it, relres
        stats = {
            "n_matvec": int(n_matvec),
            "t_matvec_total": float(t_matvec_total),
            "t_matvec_avg": float(t_matvec_total / max(n_matvec, 1)),
        }
        return x, it, relres, stats

    for k in range(1, maxiter + 1):
        it = k
        _matvec_in(p, Ap)
        denom = float(xp.real(backend.linalg.vdot(p, Ap)))
        if denom <= 0.0 or not math.isfinite(denom):
            raise RuntimeError(
                f"CG denominator is non-positive or non-finite (denom={denom}). "
                "A may be non-SPD numerically."
            )
        alpha = rsold / denom
        x += alpha * p
        r -= alpha * Ap
        rsnew = float(xp.real(backend.linalg.vdot(r, r)))
        rel = math.sqrt(rsnew) / norm_b
        if rel < tol:
            break
        beta = rsnew / max(rsold, 1e-30)
        xp.multiply(p, beta, out=Ap)
        Ap += r
        xp.copyto(p, Ap)
        rsold = rsnew

    relres = float(backend.linalg.norm(r) / norm_b)
    if not return_stats:
        return x, it, relres
    stats = {
        "n_matvec": int(n_matvec),
        "t_matvec_total": float(t_matvec_total),
        "t_matvec_avg": float(t_matvec_total / max(n_matvec, 1)),
    }
    return x, it, relres, stats


def pcg_solve_gpu(
    backend: Any,
    matvec: Callable[[Any, Any], None],
    precond: Callable[[Any, Any], None],
    b: Any,
    op_ctx: Any,
    tol: float,
    maxiter: int,
    *,
    return_stats: bool = False,
    work_prefix: str = "pcg",
) -> tuple[Any, int, float] | tuple[Any, int, float, dict[str, float]]:
    xp = backend.xp
    dtype = xp.complex128
    b = xp.asarray(b, dtype=dtype).reshape(-1)
    n = int(b.size)

    x = _ensure_workspace_vector(op_ctx, xp, f"{work_prefix}_x", n, dtype)
    r = _ensure_workspace_vector(op_ctx, xp, f"{work_prefix}_r", n, dtype)
    p = _ensure_workspace_vector(op_ctx, xp, f"{work_prefix}_p", n, dtype)
    Ap = _ensure_workspace_vector(op_ctx, xp, f"{work_prefix}_ap", n, dtype)
    z = _ensure_workspace_vector(op_ctx, xp, f"{work_prefix}_z", n, dtype)

    t_matvec_total = 0.0
    t_precond_total = 0.0
    n_matvec = 0
    n_precond = 0

    def _matvec_in(v: Any, out: Any) -> None:
        nonlocal t_matvec_total, n_matvec
        _sync_device(xp)
        t0 = time.perf_counter()
        matvec(v, out)
        _sync_device(xp)
        t_matvec_total += time.perf_counter() - t0
        n_matvec += 1

    def _precond_in(v: Any, out: Any) -> None:
        nonlocal t_precond_total, n_precond
        _sync_device(xp)
        t0 = time.perf_counter()
        precond(v, out)
        _sync_device(xp)
        t_precond_total += time.perf_counter() - t0
        n_precond += 1

    x.fill(0)
    _matvec_in(x, Ap)
    xp.subtract(b, Ap, out=r)
    _precond_in(r, z)
    xp.copyto(p, z)

    rzold = float(xp.real(backend.linalg.vdot(r, z)))
    norm_b = max(float(backend.linalg.norm(b)), 1e-30)
    it = 0

    if maxiter <= 0:
        relres = float(backend.linalg.norm(r) / norm_b)
        if not return_stats:
            return x, it, relres
        stats = {
            "n_matvec": int(n_matvec),
            "t_matvec_total": float(t_matvec_total),
            "t_matvec_avg": float(t_matvec_total / max(n_matvec, 1)),
            "n_precond": int(n_precond),
            "t_precond_total": float(t_precond_total),
            "t_precond_avg": float(t_precond_total / max(n_precond, 1)),
        }
        return x, it, relres, stats

    for k in range(1, maxiter + 1):
        it = k
        _matvec_in(p, Ap)
        denom = float(xp.real(backend.linalg.vdot(p, Ap)))
        if denom <= 0.0 or not math.isfinite(denom):
            raise RuntimeError(
                f"PCG denominator is non-positive or non-finite (denom={denom}). "
                "A may be non-SPD numerically."
            )
        alpha = rzold / denom
        x += alpha * p
        r -= alpha * Ap
        rel = float(backend.linalg.norm(r) / norm_b)
        if rel < tol:
            break
        _precond_in(r, z)
        rznew = float(xp.real(backend.linalg.vdot(r, z)))
        beta = rznew / max(rzold, 1e-30)
        xp.multiply(p, beta, out=Ap)
        Ap += z
        xp.copyto(p, Ap)
        rzold = rznew

    relres = float(backend.linalg.norm(r) / norm_b)
    if not return_stats:
        return x, it, relres
    stats = {
        "n_matvec": int(n_matvec),
        "t_matvec_total": float(t_matvec_total),
        "t_matvec_avg": float(t_matvec_total / max(n_matvec, 1)),
        "n_precond": int(n_precond),
        "t_precond_total": float(t_precond_total),
        "t_precond_avg": float(t_precond_total / max(n_precond, 1)),
    }
    return x, it, relres, stats
