from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Optional

import numpy as np

__all__ = [
    "cupy_eigsh",
    "rand_subspace_rr",
]


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _apply_matvec_block(
    matvec: Optional[Callable[[Any], Any]],
    matvec_block: Optional[Callable[[Any], Any]],
    V: Any,
    xp: Any,
) -> Any:
    if matvec_block is not None:
        return matvec_block(V)
    if matvec is None:
        raise TypeError("matvec_block is required when matvec is None.")
    cols = [matvec(V[:, i]) for i in range(int(V.shape[1]))]
    return xp.stack(cols, axis=1)


def _make_highp_block_matvec(
    matvec: Optional[Callable[[Any], Any]],
    matvec_block: Optional[Callable[[Any], Any]],
    xp: Any,
    cfg: Any = None,
) -> Callable[[Any], Any]:
    explicit = _cfg_get(cfg, "matvec_block_highp", None)
    if explicit is not None:
        def _highp(V: Any) -> Any:
            out = explicit(xp.asarray(V, dtype=xp.complex128))
            if out is None:
                raise TypeError("matvec_block_highp returned None")
            return xp.asarray(out, dtype=xp.complex128)

        return _highp

    def _default(V: Any) -> Any:
        Vh = xp.asarray(V, dtype=xp.complex128)
        out = _apply_matvec_block(matvec, matvec_block, Vh, xp)
        if out is None:
            raise TypeError("matvec_block returned None")
        return xp.asarray(out, dtype=xp.complex128)

    return _default


def _randn(xp: Any, seed: int, shape: tuple[int, ...]) -> Any:
    try:
        rng = xp.random.RandomState(seed)
        return rng.standard_normal(shape)
    except Exception:
        return xp.random.standard_normal(shape)


def _sort_eigpairs(eigvals: Any, eigvecs: Any, xp: Any) -> tuple[Any, Any]:
    idx = xp.argsort(xp.real(eigvals))[::-1]
    return xp.real(eigvals[idx]), eigvecs[:, idx]


def _prepare_init_basis(
    xp: Any, size: int, block: int, seed: int, dtype: Any, init_Q: Any = None
) -> Any:
    if init_Q is None:
        Q0 = _randn(xp, seed, (size, block)).astype(dtype)
    else:
        Q0 = xp.asarray(init_Q)
        if Q0.ndim == 1:
            Q0 = Q0.reshape(-1, 1)
        if int(Q0.shape[0]) != int(size):
            raise ValueError(f"init_Q has wrong row count: {Q0.shape[0]} != {size}")
        if int(Q0.shape[1]) < int(block):
            extra = _randn(xp, seed + 97, (size, int(block) - int(Q0.shape[1]))).astype(Q0.dtype)
            Q0 = xp.concatenate([Q0, extra], axis=1)
        elif int(Q0.shape[1]) > int(block):
            Q0 = Q0[:, : int(block)]
    Q64 = xp.asarray(Q0, dtype=xp.complex128)
    Q64, _ = xp.linalg.qr(Q64, mode="reduced")
    return xp.asarray(Q64[:, : int(block)], dtype=dtype)


def _run_rr_on_basis(
    matvec_block: Callable[[Any], Any],
    Q: Any,
    k: int,
    xp: Any,
    assume_orthonormal: bool = False,
) -> SimpleNamespace:
    Q = xp.asarray(Q, dtype=xp.complex128)
    if not assume_orthonormal:
        Q, _ = xp.linalg.qr(Q, mode="reduced")
    AQ = matvec_block(Q)
    B = Q.conj().T @ AQ
    B = 0.5 * (B + B.conj().T)
    eigvals, eigvecs_small = xp.linalg.eigh(B)
    eigvals = xp.real(eigvals)
    idx = xp.argsort(eigvals)[::-1]
    eigvals = eigvals[idx][:k]
    eigvecs = Q @ eigvecs_small[:, idx][:, :k]
    AU = AQ @ eigvecs_small[:, idx][:, :k]
    return SimpleNamespace(values=eigvals, vectors=eigvecs, Av=AU)


def _rr_residual_per_vec(
    matvec_block: Callable[[Any], Any], eigvals: Any, eigvecs: Any, xp: Any, AU: Any = None
) -> Any:
    lam = xp.asarray(eigvals, dtype=xp.complex128).reshape(1, -1)
    U = xp.asarray(eigvecs, dtype=xp.complex128)
    if AU is None:
        AU = matvec_block(U)
    else:
        AU = xp.asarray(AU, dtype=xp.complex128)
    R = AU - U * lam
    nr = xp.linalg.norm(R, axis=0)
    na = xp.maximum(xp.linalg.norm(AU, axis=0), 1e-30)
    return xp.asarray(nr / na, dtype=xp.float64)


def _residual_converged(
    res_vec: Any, k: int, tol_val: float, xp: Any, q: float = 0.90, max_factor: float = 5.0
) -> bool:
    kk = min(int(k), int(res_vec.shape[0]))
    if kk <= 0:
        return False
    head = xp.asarray(res_vec[:kk], dtype=xp.float64)
    mx = float(xp.max(head))
    if not np.isfinite(mx):
        return False
    hs = xp.sort(head)
    qi = int(max(0, min(kk - 1, round(float(q) * (kk - 1)))))
    qv = float(hs[qi])
    return (qv <= float(tol_val)) and (mx <= float(max_factor) * float(tol_val))


def _sanitize_subspace_dims(
    size: int, top_q: int, block_size: Optional[int], oversample: int, min_over: int = 16
) -> tuple[int, int, int, int]:
    n = int(size)
    k = int(top_q) + 1
    if k < 1:
        raise ValueError("top_q must be >= 0.")
    if k >= n:
        raise ValueError(f"top_q is too large: k={k}, size={n}")

    over = max(int(oversample or 0), int(min_over))
    if block_size is None:
        block = k + over
    else:
        block = int(block_size)

    block = max(block, k)
    block = min(block, n)
    rr_rank = min(k + over, block)
    return k, over, block, rr_rank


def _run_rr_check(
    matvec_block_highp: Callable[[Any], Any],
    Q: Any,
    k: int,
    xp: Any,
    step: int,
    n_iter: int,
    tol_val: float,
    rr_every: int,
    residual_every: int,
    rr_warmup_iters: int,
    q_res: float,
    max_factor: float,
) -> tuple[Optional[SimpleNamespace], bool]:
    should_rr = ((step >= rr_warmup_iters) and (step % rr_every == 0)) or (step == n_iter)
    if not should_rr:
        return None, False

    Qrr = xp.asarray(Q, dtype=xp.complex128)
    rr = _run_rr_on_basis(matvec_block_highp, Qrr, k, xp, assume_orthonormal=True)
    should_residual = (step == n_iter) or (
        residual_every > 0 and step >= rr_warmup_iters and step % residual_every == 0
    )
    converged = False
    if should_residual:
        res = _rr_residual_per_vec(
            matvec_block_highp,
            rr.values,
            rr.vectors,
            xp,
            AU=getattr(rr, "Av", None),
        )
        rr.residual = res
        converged = _residual_converged(
            res,
            k,
            tol_val,
            xp,
            q=q_res,
            max_factor=max_factor,
        )
    return rr, converged


def _attach_solver_stats(result: Any, **stats: Any) -> Any:
    if result is None:
        return None
    for key, value in stats.items():
        setattr(result, key, value)
    return result


def rand_subspace_rr(
    matvec: Optional[Callable[[Any], Any]],
    size: int,
    top_q: int,
    matvec_block: Optional[Callable[[Any], Any]] = None,
    block_size: Optional[int] = None,
    oversample: int = 2,
    tol: float = 1e-6,
    maxiter: Optional[int] = None,
    xp: Any = None,
    cfg: Any = None,
) -> SimpleNamespace:
    xp = np if xp is None else xp
    min_over = int(_cfg_get(cfg, "min_oversample", 16))
    k, _, block, _ = _sanitize_subspace_dims(
        size=size,
        top_q=top_q,
        block_size=block_size,
        oversample=oversample,
        min_over=min_over,
    )
    n_iter = int(_cfg_get(cfg, "maxiter", 3 if maxiter is None else maxiter))
    tol_val = float(_cfg_get(cfg, "tol", tol))
    rr_every = max(int(_cfg_get(cfg, "rr_check_every", 1)), 1)
    residual_every = max(int(_cfg_get(cfg, "residual_check_every", 1)), 1)
    rr_warmup_iters = max(int(_cfg_get(cfg, "rr_warmup_iters", 1)), 1)
    q_res = float(_cfg_get(cfg, "residual_q", 0.90))
    max_factor = float(_cfg_get(cfg, "residual_max_factor", 5.0))
    init_Q = _cfg_get(cfg, "init_Q", None)
    matvec_block_highp = _make_highp_block_matvec(matvec, matvec_block, xp, cfg)

    Q = _prepare_init_basis(xp, size, block, 1, xp.complex128, init_Q=init_Q)
    rr = None
    for it in range(max(n_iter, 1)):
        step = it + 1
        Y = matvec_block_highp(Q)
        Q, _ = xp.linalg.qr(Y, mode="reduced")
        rr, converged = _run_rr_check(
            matvec_block_highp,
            Q,
            k,
            xp,
            step,
            n_iter,
            tol_val,
            rr_every,
            residual_every,
            rr_warmup_iters,
            q_res,
            max_factor,
        )
        if rr is not None and converged:
            return rr

    if rr is None:
        rr = _run_rr_on_basis(matvec_block_highp, Q, k, xp, assume_orthonormal=True)
    return rr


def _build_eigsh_start_vector(
    matvec_block_highp: Callable[[Any], Any], size: int, xp: Any, cfg: Any = None
) -> Any:
    strategy = str(_cfg_get(cfg, "warm_start_strategy", "none") or "none").lower()
    if strategy == "none":
        return None
    if strategy == "init_q_first_col":
        init_Q = _cfg_get(cfg, "init_Q", None)
        if init_Q is None:
            return None
        v0 = xp.asarray(init_Q, dtype=xp.complex128)
        if v0.ndim == 2:
            v0 = v0[:, 0]
        return xp.asarray(v0, dtype=xp.complex128).reshape(-1)

    v0 = xp.asarray(_randn(xp, 41, (size,)), dtype=xp.complex128).reshape(-1)
    nv = float(xp.linalg.norm(v0))
    if nv > 0.0:
        v0 = v0 / nv
    if strategy == "random":
        return v0
    if strategy == "power1":
        v1 = matvec_block_highp(v0.reshape(-1, 1))[:, 0]
        n1 = float(xp.linalg.norm(v1))
        if n1 > 0.0:
            v1 = v1 / n1
        return xp.asarray(v1, dtype=xp.complex128).reshape(-1)
    raise ValueError(f"unknown warm_start_strategy: {strategy}")


def cupy_eigsh(
    matvec: Optional[Callable[[Any], Any]],
    size: int,
    top_q: int,
    matvec_block: Optional[Callable[[Any], Any]] = None,
    block_size: Optional[int] = None,
    oversample: int = 2,
    tol: float = 1e-6,
    maxiter: Optional[int] = None,
    xp: Any = None,
    cfg: Any = None,
) -> SimpleNamespace:
    xp = np if xp is None else xp
    if xp is np:
        raise RuntimeError("cupy_eigsh expects GPU backend.xp (cupy).")
    try:
        import cupyx.scipy.sparse.linalg as csl
    except Exception as exc:
        raise RuntimeError("cupyx.scipy.sparse.linalg is required for eigsh") from exc

    min_over = int(_cfg_get(cfg, "min_oversample", 16))
    k, _, _ncv_hint, _ = _sanitize_subspace_dims(
        size=size,
        top_q=top_q,
        block_size=block_size,
        oversample=oversample,
        min_over=min_over,
    )
    matvec_block_highp = _make_highp_block_matvec(matvec, matvec_block, xp, cfg)
    op_dtype = xp.complex128

    def _eigsh_mv(v: Any) -> Any:
        vv = xp.asarray(v, dtype=op_dtype).reshape(-1, 1)
        out = matvec_block_highp(vv)
        return xp.asarray(out[:, 0], dtype=op_dtype).reshape(-1)

    def _eigsh_mm(V: Any) -> Any:
        vV = xp.asarray(V, dtype=op_dtype)
        return xp.asarray(matvec_block_highp(vV), dtype=op_dtype)

    a_op = csl.LinearOperator(
        (size, size), matvec=_eigsh_mv, matmat=_eigsh_mm, dtype=op_dtype
    )
    ncv = _cfg_get(cfg, "ncv", None)
    if ncv is None:
        ncv = max(2 * int(k) + 32, int(k) + 2)
    ncv = max(int(k) + 2, int(ncv))
    ncv = min(int(size) - 1, int(ncv))
    maxiter_eff = _cfg_get(cfg, "maxiter", maxiter)
    v0 = _build_eigsh_start_vector(matvec_block_highp, size, xp, cfg)

    eigvals, eigvecs = csl.eigsh(
        a_op,
        k=int(k),
        which=str(_cfg_get(cfg, "which", "LA")),
        v0=v0,
        ncv=ncv,
        maxiter=maxiter_eff,
        tol=float(_cfg_get(cfg, "tol", tol)),
        return_eigenvectors=True,
    )
    eigvals, eigvecs = _sort_eigpairs(eigvals, eigvecs, xp)
    return SimpleNamespace(values=eigvals[:k], vectors=eigvecs[:, :k])
