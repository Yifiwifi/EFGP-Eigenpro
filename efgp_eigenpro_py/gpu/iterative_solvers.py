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
    profile_components: bool = True,
    spd_guard: bool = True,
    trace_callback: Optional[Callable[[dict[str, Any]], None]] = None,
) -> tuple[Any, int, float] | tuple[Any, int, float, dict[str, float]]:
    """
    Conjugate Gradient on GPU with reusable workspaces.

    ``spd_guard=True`` (default) raises if ``p* A p <= 0`` or non-finite, which
    flags a numerically non-SPD operator.  ``spd_guard=False`` is for *consistent
    singular SPSD* systems (e.g. the deflated operator ``H = P_D A``): a
    non-positive / non-finite denominator is treated as a Krylov breakdown that is
    only acceptable once the residual already meets ``tol``; otherwise it raises so
    failures are never silent.
    """
    xp = backend.xp
    solve_dtype = getattr(op_ctx, "solve_dtype", None)
    if solve_dtype is None:
        solve_dtype = getattr(backend, "dtype", None)
    b = xp.asarray(b).reshape(-1)
    dtype = xp.dtype(b.dtype if solve_dtype is None else solve_dtype)
    if dtype.kind in ("b", "i", "u"):
        dtype = xp.dtype(xp.float64)
    elif dtype == xp.dtype(xp.float16):
        dtype = xp.dtype(xp.float32)
    b = b.astype(dtype, copy=False)
    n = int(b.size)

    x = _ensure_workspace_vector(op_ctx, xp, f"{work_prefix}_x", n, dtype)
    r = _ensure_workspace_vector(op_ctx, xp, f"{work_prefix}_r", n, dtype)
    p = _ensure_workspace_vector(op_ctx, xp, f"{work_prefix}_p", n, dtype)
    Ap = _ensure_workspace_vector(op_ctx, xp, f"{work_prefix}_ap", n, dtype)

    t_matvec_total = 0.0
    n_matvec = 0
    trace_t0 = time.perf_counter()

    def _matvec_in(v: Any, out: Any) -> None:
        nonlocal t_matvec_total, n_matvec
        if profile_components:
            _sync_device(xp)
        t0 = time.perf_counter()
        matvec(v, out)
        if profile_components:
            _sync_device(xp)
        t_matvec_total += time.perf_counter() - t0
        n_matvec += 1

    if not profile_components:
        _sync_device(xp)
    x.fill(0)
    _matvec_in(x, Ap)
    xp.subtract(b, Ap, out=r)
    xp.copyto(p, r)

    rsold = float(xp.real(backend.linalg.vdot(r, r)))
    norm_b = max(float(backend.linalg.norm(b)), 1e-30)
    it = 0
    if trace_callback is not None:
        trace_callback(
            {
                "iteration": 0,
                "relres": math.sqrt(max(rsold, 0.0)) / norm_b,
                "elapsed_time": time.perf_counter() - trace_t0,
                "x": x,
                "is_final": False,
                "converged": False,
            }
        )

    if maxiter <= 0:
        relres = float(backend.linalg.norm(r) / norm_b)
        if not profile_components:
            _sync_device(xp)
        if not return_stats:
            return x, it, relres
        matvec_total = t_matvec_total if profile_components else float("nan")
        stats = {
            "n_matvec": int(n_matvec),
            "t_matvec_total": float(matvec_total),
            "t_matvec_avg": float(matvec_total / max(n_matvec, 1)),
            "profile_components": bool(profile_components),
        }
        return x, it, relres, stats

    status = "running"
    max_imag_ratio = 0.0
    for k in range(1, maxiter + 1):
        it = k
        _matvec_in(p, Ap)
        pAp = backend.linalg.vdot(p, Ap)
        denom = float(xp.real(pAp))
        if not spd_guard:
            imag = abs(float(xp.imag(pAp)))
            ratio = imag / (abs(denom) + 1e-30)
            if ratio > max_imag_ratio:
                max_imag_ratio = ratio
        if denom <= 0.0 or not math.isfinite(denom):
            if spd_guard:
                raise RuntimeError(
                    f"CG denominator is non-positive or non-finite (denom={denom}). "
                    "A may be non-SPD numerically."
                )
            # Consistent SPSD: only accept breakdown if already converged.
            rel_now = math.sqrt(max(rsold, 0.0)) / norm_b
            if rel_now <= tol:
                status = "converged_spsd_breakdown"
                break
            raise RuntimeError(
                f"deflated CG breakdown (denom={denom}) but residual not converged "
                f"(rel={rel_now:.3e} > tol={tol:.3e}). Projector/recovery may be wrong."
            )
        alpha = rsold / denom
        x += alpha * p
        r -= alpha * Ap
        rsnew = float(xp.real(backend.linalg.vdot(r, r)))
        rel = math.sqrt(rsnew) / norm_b
        converged = rel < tol
        if trace_callback is not None:
            trace_callback(
                {
                    "iteration": it,
                    "relres": rel,
                    "elapsed_time": time.perf_counter() - trace_t0,
                    "x": x,
                    "is_final": converged or it >= maxiter,
                    "converged": converged,
                }
            )
        if converged:
            status = "converged"
            break
        beta = rsnew / max(rsold, 1e-30)
        p *= beta
        p += r
        rsold = rsnew
    else:
        status = "maxiter"

    relres = float(backend.linalg.norm(r) / norm_b)
    if not profile_components:
        _sync_device(xp)
    if not return_stats:
        return x, it, relres
    matvec_total = t_matvec_total if profile_components else float("nan")
    stats = {
        "n_matvec": int(n_matvec),
        "t_matvec_total": float(matvec_total),
        "t_matvec_avg": float(matvec_total / max(n_matvec, 1)),
        "profile_components": bool(profile_components),
        "status": status,
        "max_imag_ratio": float(max_imag_ratio),
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
    profile_components: bool = True,
    spd_guard: bool = True,
    trace_callback: Optional[Callable[[dict[str, Any]], None]] = None,
) -> tuple[Any, int, float] | tuple[Any, int, float, dict[str, float]]:
    """
    Preconditioned CG on GPU.

    ``spd_guard`` behaves like in :func:`cg_solve_gpu`: ``True`` raises on a
    non-positive / non-finite ``p* A p``; ``False`` (deflated-PCG / DPCG on a
    consistent singular SPSD operator) only accepts a breakdown once the residual
    already meets ``tol``, otherwise it raises so failures are never silent.
    """
    xp = backend.xp
    solve_dtype = getattr(op_ctx, "solve_dtype", None)
    if solve_dtype is None:
        solve_dtype = getattr(backend, "dtype", None)
    b = xp.asarray(b).reshape(-1)
    dtype = xp.dtype(b.dtype if solve_dtype is None else solve_dtype)
    if dtype.kind in ("b", "i", "u"):
        dtype = xp.dtype(xp.float64)
    elif dtype == xp.dtype(xp.float16):
        dtype = xp.dtype(xp.float32)
    b = b.astype(dtype, copy=False)
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
    trace_t0 = time.perf_counter()

    def _matvec_in(v: Any, out: Any) -> None:
        nonlocal t_matvec_total, n_matvec
        if profile_components:
            _sync_device(xp)
        t0 = time.perf_counter()
        matvec(v, out)
        if profile_components:
            _sync_device(xp)
        t_matvec_total += time.perf_counter() - t0
        n_matvec += 1

    def _precond_in(v: Any, out: Any) -> None:
        nonlocal t_precond_total, n_precond
        if profile_components:
            _sync_device(xp)
        t0 = time.perf_counter()
        precond(v, out)
        if profile_components:
            _sync_device(xp)
        t_precond_total += time.perf_counter() - t0
        n_precond += 1

    if not profile_components:
        _sync_device(xp)
    x.fill(0)
    _matvec_in(x, Ap)
    xp.subtract(b, Ap, out=r)
    _precond_in(r, z)
    xp.copyto(p, z)

    rzold = float(xp.real(backend.linalg.vdot(r, z)))
    norm_b = max(float(backend.linalg.norm(b)), 1e-30)
    it = 0
    if trace_callback is not None:
        trace_callback(
            {
                "iteration": 0,
                "relres": float(backend.linalg.norm(r) / norm_b),
                "elapsed_time": time.perf_counter() - trace_t0,
                "x": x,
                "is_final": False,
                "converged": False,
            }
        )

    if maxiter <= 0:
        relres = float(backend.linalg.norm(r) / norm_b)
        if not profile_components:
            _sync_device(xp)
        if not return_stats:
            return x, it, relres
        matvec_total = t_matvec_total if profile_components else float("nan")
        precond_total = t_precond_total if profile_components else float("nan")
        stats = {
            "n_matvec": int(n_matvec),
            "t_matvec_total": float(matvec_total),
            "t_matvec_avg": float(matvec_total / max(n_matvec, 1)),
            "n_precond": int(n_precond),
            "t_precond_total": float(precond_total),
            "t_precond_avg": float(precond_total / max(n_precond, 1)),
            "profile_components": bool(profile_components),
        }
        return x, it, relres, stats

    status = "running"
    max_imag_ratio = 0.0
    for k in range(1, maxiter + 1):
        it = k
        _matvec_in(p, Ap)
        pAp = backend.linalg.vdot(p, Ap)
        denom = float(xp.real(pAp))
        if not spd_guard:
            imag = abs(float(xp.imag(pAp)))
            ratio = imag / (abs(denom) + 1e-30)
            if ratio > max_imag_ratio:
                max_imag_ratio = ratio
        if denom <= 0.0 or not math.isfinite(denom):
            if spd_guard:
                raise RuntimeError(
                    f"PCG denominator is non-positive or non-finite (denom={denom}). "
                    "A may be non-SPD numerically."
                )
            rel_now = float(backend.linalg.norm(r) / norm_b)
            if rel_now <= tol:
                status = "converged_spsd_breakdown"
                break
            raise RuntimeError(
                f"deflated PCG breakdown (denom={denom}) but residual not converged "
                f"(rel={rel_now:.3e} > tol={tol:.3e}). Projector/recovery may be wrong."
            )
        alpha = rzold / denom
        x += alpha * p
        r -= alpha * Ap
        rrnew = float(xp.real(backend.linalg.vdot(r, r)))
        rel = math.sqrt(max(rrnew, 0.0)) / norm_b 
        converged = rel < tol
        if trace_callback is not None:
            trace_callback(
                {
                    "iteration": it,
                    "relres": rel,
                    "elapsed_time": time.perf_counter() - trace_t0,
                    "x": x,
                    "is_final": converged or it >= maxiter,
                    "converged": converged,
                }
            )
        if converged:
            status = "converged"
            break
        _precond_in(r, z)
        rznew = float(xp.real(backend.linalg.vdot(r, z)))
        beta = rznew / max(rzold, 1e-30)
        p *= beta
        p += z
        rzold = rznew
    else:
        status = "maxiter"

    relres = float(backend.linalg.norm(r) / norm_b)
    if not profile_components:
        _sync_device(xp)
    if not return_stats:
        return x, it, relres
    matvec_total = t_matvec_total if profile_components else float("nan")
    precond_total = t_precond_total if profile_components else float("nan")
    stats = {
        "n_matvec": int(n_matvec),
        "t_matvec_total": float(matvec_total),
        "t_matvec_avg": float(matvec_total / max(n_matvec, 1)),
        "n_precond": int(n_precond),
        "t_precond_total": float(precond_total),
        "t_precond_avg": float(precond_total / max(n_precond, 1)),
        "profile_components": bool(profile_components),
        "status": status,
        "max_imag_ratio": float(max_imag_ratio),
    }
    return x, it, relres, stats


def fgmres_solve_gpu(
    backend: Any,
    matvec: Callable[[Any, Any], None],
    precond: Callable[[Any, Any], None],
    b: Any,
    op_ctx: Any,
    tol: float,
    maxiter: int,
    *,
    restart: int = 50,
    return_stats: bool = False,
    work_prefix: str = "fgmres",
    profile_components: bool = True,
) -> tuple[Any, int, float] | tuple[Any, int, float, dict[str, float]]:
    """
    Flexible GMRES on GPU with reusable workspaces.

    The preconditioner may vary between iterations. We use a right-preconditioned
    FGMRES formulation where each Krylov basis vector ``v_k`` is preconditioned
    into ``z_k = M_k^{-1} v_k`` before applying ``A z_k``.
    """
    xp = backend.xp
    solve_dtype = getattr(op_ctx, "solve_dtype", None)
    if solve_dtype is None:
        solve_dtype = getattr(backend, "dtype", None)
    b = xp.asarray(b).reshape(-1)
    dtype = xp.dtype(b.dtype if solve_dtype is None else solve_dtype)
    if dtype.kind in ("b", "i", "u"):
        dtype = xp.dtype(xp.float64)
    elif dtype == xp.dtype(xp.float16):
        dtype = xp.dtype(xp.float32)
    b = b.astype(dtype, copy=False)
    n = int(b.size)
    restart = max(1, min(int(restart), int(maxiter) if int(maxiter) > 0 else 1))

    x = _ensure_workspace_vector(op_ctx, xp, f"{work_prefix}_x", n, dtype)
    r = _ensure_workspace_vector(op_ctx, xp, f"{work_prefix}_r", n, dtype)
    Ax = _ensure_workspace_vector(op_ctx, xp, f"{work_prefix}_ax", n, dtype)
    w = _ensure_workspace_vector(op_ctx, xp, f"{work_prefix}_w", n, dtype)
    tmp = _ensure_workspace_vector(op_ctx, xp, f"{work_prefix}_tmp", n, dtype)
    V = getattr(op_ctx, f"{work_prefix}_V", None)
    if V is None or tuple(getattr(V, "shape", ())) != (n, restart + 1) or V.dtype != dtype:
        V = xp.empty((n, restart + 1), dtype=dtype)
        setattr(op_ctx, f"{work_prefix}_V", V)
    Z = getattr(op_ctx, f"{work_prefix}_Z", None)
    if Z is None or tuple(getattr(Z, "shape", ())) != (n, restart) or Z.dtype != dtype:
        Z = xp.empty((n, restart), dtype=dtype)
        setattr(op_ctx, f"{work_prefix}_Z", Z)
    H = getattr(op_ctx, f"{work_prefix}_H", None)
    if H is None or tuple(getattr(H, "shape", ())) != (restart + 1, restart) or H.dtype != dtype:
        H = xp.zeros((restart + 1, restart), dtype=dtype)
        setattr(op_ctx, f"{work_prefix}_H", H)

    t_matvec_total = 0.0
    t_precond_total = 0.0
    n_matvec = 0
    n_precond = 0

    def _matvec_in(v: Any, out: Any) -> None:
        nonlocal t_matvec_total, n_matvec
        if profile_components:
            _sync_device(xp)
        t0 = time.perf_counter()
        matvec(v, out)
        if profile_components:
            _sync_device(xp)
        t_matvec_total += time.perf_counter() - t0
        n_matvec += 1

    def _precond_in(v: Any, out: Any) -> None:
        nonlocal t_precond_total, n_precond
        if profile_components:
            _sync_device(xp)
        t0 = time.perf_counter()
        precond(v, out)
        if profile_components:
            _sync_device(xp)
        t_precond_total += time.perf_counter() - t0
        n_precond += 1

    if not profile_components:
        _sync_device(xp)
    x.fill(0)
    _matvec_in(x, Ax)
    xp.subtract(b, Ax, out=r)
    norm_b = max(float(backend.linalg.norm(b)), 1e-30)
    beta = float(backend.linalg.norm(r))
    relres = beta / norm_b
    it = 0
    status = "running"

    if maxiter <= 0 or relres <= tol:
        if relres <= tol:
            status = "converged"
        if not profile_components:
            _sync_device(xp)
        if not return_stats:
            return x, it, relres
        matvec_total = t_matvec_total if profile_components else float("nan")
        precond_total = t_precond_total if profile_components else float("nan")
        stats = {
            "n_matvec": int(n_matvec),
            "t_matvec_total": float(matvec_total),
            "t_matvec_avg": float(matvec_total / max(n_matvec, 1)),
            "n_precond": int(n_precond),
            "t_precond_total": float(precond_total),
            "t_precond_avg": float(precond_total / max(n_precond, 1)),
            "profile_components": bool(profile_components),
            "status": status,
            "restart": int(restart),
        }
        return x, it, relres, stats

    rhs = xp.zeros((restart + 1,), dtype=dtype)
    while it < int(maxiter):
        H.fill(0)
        rhs.fill(0)
        beta = float(backend.linalg.norm(r))
        if beta <= tol * norm_b:
            status = "converged"
            break
        xp.copyto(V[:, 0], r / beta)
        rhs[0] = beta

        inner_used = 0
        for j in range(restart):
            if it >= int(maxiter):
                status = "maxiter"
                break
            _precond_in(V[:, j], Z[:, j])
            _matvec_in(Z[:, j], w)
            for i in range(j + 1):
                hij = backend.linalg.vdot(V[:, i], w)
                H[i, j] = hij
                w -= hij * V[:, i]
            hnext = float(backend.linalg.norm(w))
            H[j + 1, j] = hnext
            inner_used = j + 1
            if hnext > 0.0:
                xp.copyto(V[:, j + 1], w / hnext)
            Hsub = H[: j + 2, : j + 1]
            y, *_ = xp.linalg.lstsq(Hsub, rhs[: j + 2], rcond=None)
            xp.copyto(tmp, Z[:, : j + 1] @ y)
            rel_candidate = float(backend.linalg.norm(rhs[: j + 2] - Hsub @ y) / norm_b)
            it += 1
            if rel_candidate <= tol:
                x += tmp
                relres = rel_candidate
                status = "converged"
                break
            if hnext <= 1e-30:
                x += tmp
                _matvec_in(x, Ax)
                xp.subtract(b, Ax, out=r)
                relres = float(backend.linalg.norm(r) / norm_b)
                status = "converged_happy_breakdown" if relres <= tol else "happy_breakdown"
                break
        else:
            j = inner_used - 1

        if status in ("converged", "converged_happy_breakdown", "happy_breakdown"):
            break
        if inner_used <= 0:
            status = "maxiter"
            break

        y, *_ = xp.linalg.lstsq(H[: inner_used + 1, :inner_used], rhs[: inner_used + 1], rcond=None)
        x += Z[:, :inner_used] @ y
        _matvec_in(x, Ax)
        xp.subtract(b, Ax, out=r)
        relres = float(backend.linalg.norm(r) / norm_b)
        if relres <= tol:
            status = "converged"
            break
        if it >= int(maxiter):
            status = "maxiter"
            break

    if not profile_components:
        _sync_device(xp)
    if not return_stats:
        return x, it, relres
    matvec_total = t_matvec_total if profile_components else float("nan")
    precond_total = t_precond_total if profile_components else float("nan")
    stats = {
        "n_matvec": int(n_matvec),
        "t_matvec_total": float(matvec_total),
        "t_matvec_avg": float(matvec_total / max(n_matvec, 1)),
        "n_precond": int(n_precond),
        "t_precond_total": float(precond_total),
        "t_precond_avg": float(precond_total / max(n_precond, 1)),
        "profile_components": bool(profile_components),
        "status": status,
        "restart": int(restart),
    }
    return x, it, relres, stats
