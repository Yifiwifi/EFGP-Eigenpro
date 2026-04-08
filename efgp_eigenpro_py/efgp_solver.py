from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np

from .discretization import GridSpec, basis_weights, choose_grid_params, generate_multi_index
from .eigenspace import EigenPairs, estimate_top_eigenspace
from .eigenpro_precond import EigenProPreconditioner, build_preconditioner
from .linear_solvers import pcg, richardson
from .nufft_ops import (
    nufft1d1,
    nufft1d2,
    nufft2d1,
    nufft2d2,
    nufft3d1,
    nufft3d2,
    nufftnd1,
    nufftnd2,
)
from .toeplitz import (
    ToeplitzWorkspace,
    ToeplitzBlockWorkspace,
    _HAS_SCIPY_FFT,
    _fftn,
    _ifftn,
    apply_toeplitz_fft_nd,
    apply_toeplitz_fft_nd_block,
    make_toeplitz_workspace,
    make_toeplitz_block_workspace,
)
from .kernels import KernelSpec


@dataclass
class PrecomputeState:
    grid: GridSpec
    weights: np.ndarray #diag(weights)=D
    Gf: np.ndarray      # \hat{v}, representing F^H F
                        # with D and v, we can reconstruct A
    rhs: np.ndarray     #\Phi^*y
    x_center: np.ndarray
    toeplitz_ws: Optional[ToeplitzWorkspace] = None
    toeplitz_ws_block: Optional[ToeplitzBlockWorkspace] = None
    apply_w: Optional[np.ndarray] = None
    apply_w_block: Optional[np.ndarray] = None
    multi_index: Optional[np.ndarray] = None
    multi_index: Optional[np.ndarray] = None


class EFGPSolver:
    """
    EFGP weight-space solver skeleton with optional EigenPro preconditioning.
    """
    def __init__(
        self,
        kernel: KernelSpec,
        reg_lambda: float,
        eps: float,
        nufft_tol: float,
        *,
        l2scaled: bool = False,
    ) -> None:
        self.kernel = kernel
        self.reg_lambda = reg_lambda
        self.eps = eps
        self.nufft_tol = nufft_tol
        self.l2scaled = bool(l2scaled)

    def _center_and_scale(self, x: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
        x0 = np.min(x, axis=0)
        x1 = np.max(x, axis=0)
        x_center = (x0 + x1) / 2.0
        tphx = 2.0 * np.pi * h * (x - x_center)
        # Avoid exact +pi which FINUFFT may fold ambiguously.
        tphx = np.clip(tphx, -np.pi, np.nextafter(np.pi, 0.0))
        return tphx, x_center

    def _scale_from_center(self, x: np.ndarray, x_center: np.ndarray, h: float) -> np.ndarray:
        tphx = 2.0 * np.pi * h * (x - x_center)
        # Avoid exact +pi which FINUFFT may fold ambiguously.
        return np.clip(tphx, -np.pi, np.nextafter(np.pi, 0.0))

    def _ensure_2d(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array.")
        if x.shape[1] != self.kernel.dim:
            raise ValueError("x must have shape (n, dim).")
        return x

    def precompute(self, x: np.ndarray, y: np.ndarray) -> PrecomputeState:
        """
        Precompute Toeplitz FFT embedding and right-hand side.
        """
        x = self._ensure_2d(x)
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.reshape(-1)
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if y.shape[0] != x.shape[0]:
            raise ValueError("x and y must have matching lengths.")
        y = np.ascontiguousarray(y)
        L = np.max(np.max(x, axis=0) - np.min(x, axis=0))
        grid = choose_grid_params(self.kernel, self.eps, L, l2scaled=self.l2scaled)
        tphx, x_center = self._center_and_scale(x, grid.h)
        weights = np.ascontiguousarray(basis_weights(self.kernel, grid.xis, grid.h).reshape(-1))

        if self.kernel.dim == 1:
            c = np.ones(x.shape[0], dtype=complex)
            # [XtXcol, Gf] is [v,\hat{v}] in Algorithm 6 step2 of EFGP and (20)
            # rhs is b or \Phi^*y in Algorithm 6 step3 of EFGP and (20)
            XtXcol = nufft1d1(tphx[:, 0], c, 2 * grid.mtot - 1, self.nufft_tol, -1)
            Gf = _fftn(XtXcol, overwrite_x=_HAS_SCIPY_FFT)
            rhs = nufft1d1(tphx[:, 0], y, grid.mtot, self.nufft_tol, -1)
            rhs = np.multiply(weights, rhs)
        elif self.kernel.dim == 2:
            c = np.ones(x.shape[0], dtype=complex)
            XtXcol = nufft2d1(
                tphx[:, 0], tphx[:, 1], c, 2 * grid.mtot - 1, 2 * grid.mtot - 1,
                self.nufft_tol, -1
            )
            Gf = _fftn(XtXcol, overwrite_x=_HAS_SCIPY_FFT)
            rhs = nufft2d1(
                tphx[:, 0], tphx[:, 1], y, grid.mtot, grid.mtot, self.nufft_tol, -1
            )
            rhs = np.ascontiguousarray(rhs.reshape(-1))
            np.multiply(weights, rhs, out=rhs)
        elif self.kernel.dim == 3:
            c = np.ones(x.shape[0], dtype=complex)
            XtXcol = nufft3d1(
                tphx[:, 0], tphx[:, 1], tphx[:, 2], c,
                2 * grid.mtot - 1, 2 * grid.mtot - 1, 2 * grid.mtot - 1,
                self.nufft_tol, -1
            )
            Gf = _fftn(XtXcol, overwrite_x=_HAS_SCIPY_FFT)
            rhs = nufft3d1(
                tphx[:, 0], tphx[:, 1], tphx[:, 2], y,
                grid.mtot, grid.mtot, grid.mtot, self.nufft_tol, -1
            )
            rhs = np.ascontiguousarray(rhs.reshape(-1))
            np.multiply(weights, rhs, out=rhs)
        else:
            c = np.ones(x.shape[0], dtype=complex)
            ms_xtx = 2 * grid.mtot - 1
            XtXcol = nufftnd1(tphx, c, ms_xtx, self.nufft_tol, -1)
            XtXcol = XtXcol.reshape((ms_xtx,) * self.kernel.dim)
            Gf = _fftn(XtXcol, overwrite_x=_HAS_SCIPY_FFT)
            rhs = nufftnd1(tphx, y, grid.mtot, self.nufft_tol, -1)
            rhs = np.ascontiguousarray(rhs.reshape(-1))
            np.multiply(weights, rhs, out=rhs)

        Gf = np.ascontiguousarray(Gf)
        rhs = np.ascontiguousarray(rhs.reshape(-1))
        toeplitz_ws = None
        apply_w = None
        return PrecomputeState(
            grid=grid,
            weights=weights,
            Gf=Gf,
            rhs=rhs,
            x_center=x_center,
            toeplitz_ws=toeplitz_ws,
            apply_w=apply_w,
        )

    def precompute_streaming(
        self,
        chunk_iter: Iterable[tuple[np.ndarray, np.ndarray]],
        n_total: int,
        *,
        x_bounds: Optional[tuple[np.ndarray, np.ndarray]] = None,
        x_center: Optional[np.ndarray] = None,
        L: Optional[float] = None,
    ) -> PrecomputeState:
        """
        Streaming precompute for large datasets.
        chunk_iter should yield (x_chunk, y_chunk) pairs.
        Provide x_bounds=(xmin, xmax) or (x_center, L) to avoid a full pass.
        """
        if x_bounds is not None:
            x_min, x_max = x_bounds
            x_min = np.asarray(x_min, dtype=float).reshape(-1)
            x_max = np.asarray(x_max, dtype=float).reshape(-1)
            x_center = (x_min + x_max) / 2.0
            L = float(np.max(x_max - x_min))
        if x_center is None or L is None:
            raise ValueError("precompute_streaming requires x_bounds or both x_center and L.")
        x_center = np.asarray(x_center, dtype=float).reshape(-1)

        grid = choose_grid_params(self.kernel, self.eps, L, l2scaled=self.l2scaled)
        weights = np.ascontiguousarray(basis_weights(self.kernel, grid.xis, grid.h).reshape(-1))

        dim = self.kernel.dim
        ms_xtx = 2 * grid.mtot - 1
        if dim == 1:
            XtXcol_acc = np.zeros(ms_xtx, dtype=complex)
            rhs_acc = np.zeros(grid.mtot, dtype=complex)
        elif dim == 2:
            XtXcol_acc = np.zeros((ms_xtx, ms_xtx), dtype=complex)
            rhs_acc = np.zeros((grid.mtot, grid.mtot), dtype=complex)
        elif dim == 3:
            XtXcol_acc = np.zeros((ms_xtx, ms_xtx, ms_xtx), dtype=complex)
            rhs_acc = np.zeros((grid.mtot, grid.mtot, grid.mtot), dtype=complex)
        else:
            XtXcol_acc = np.zeros((ms_xtx,) * dim, dtype=complex)
            rhs_acc = np.zeros((grid.mtot,) * dim, dtype=complex)

        count = 0
        ones_cache: Optional[np.ndarray] = None
        for x_chunk, y_chunk in chunk_iter:
            x_chunk = self._ensure_2d(x_chunk)
            if x_chunk.shape[1] != dim:
                raise ValueError("x_chunk must have shape (n, dim).")
            y_chunk = np.asarray(y_chunk)
            if y_chunk.ndim == 2 and y_chunk.shape[1] == 1:
                y_chunk = y_chunk.reshape(-1)
            if y_chunk.ndim != 1:
                raise ValueError("y_chunk must be a 1D array.")
            if y_chunk.shape[0] != x_chunk.shape[0]:
                raise ValueError("x_chunk and y_chunk must have matching lengths.")
            y_chunk = np.ascontiguousarray(y_chunk)

            tphx = self._scale_from_center(x_chunk, x_center, grid.h)
            n_chunk = x_chunk.shape[0]
            if ones_cache is None or ones_cache.shape[0] < n_chunk:
                ones_cache = np.ones(n_chunk, dtype=complex)
            c = ones_cache[:n_chunk]

            if dim == 1:
                XtXcol_acc += nufft1d1(tphx[:, 0], c, ms_xtx, self.nufft_tol, -1)
                rhs_acc += nufft1d1(tphx[:, 0], y_chunk, grid.mtot, self.nufft_tol, -1)
            elif dim == 2:
                XtXcol_acc += nufft2d1(
                    tphx[:, 0], tphx[:, 1], c, ms_xtx, ms_xtx, self.nufft_tol, -1
                )
                rhs_acc += nufft2d1(
                    tphx[:, 0], tphx[:, 1], y_chunk, grid.mtot, grid.mtot, self.nufft_tol, -1
                )
            elif dim == 3:
                XtXcol_acc += nufft3d1(
                    tphx[:, 0],
                    tphx[:, 1],
                    tphx[:, 2],
                    c,
                    ms_xtx,
                    ms_xtx,
                    ms_xtx,
                    self.nufft_tol,
                    -1,
                )
                rhs_acc += nufft3d1(
                    tphx[:, 0],
                    tphx[:, 1],
                    tphx[:, 2],
                    y_chunk,
                    grid.mtot,
                    grid.mtot,
                    grid.mtot,
                    self.nufft_tol,
                    -1,
                )
            else:
                XtXcol_chunk = nufftnd1(tphx, c, ms_xtx, self.nufft_tol, -1)
                XtXcol_acc += np.asarray(XtXcol_chunk).reshape((ms_xtx,) * dim)
                rhs_chunk = nufftnd1(tphx, y_chunk, grid.mtot, self.nufft_tol, -1)
                rhs_acc += np.asarray(rhs_chunk).reshape((grid.mtot,) * dim)

            count += x_chunk.shape[0]

        if n_total is not None and count != n_total:
            raise ValueError(f"precompute_streaming saw {count} samples, expected {n_total}.")

        Gf = _fftn(XtXcol_acc, overwrite_x=_HAS_SCIPY_FFT)
        rhs = np.ascontiguousarray(rhs_acc.reshape(-1))
        np.multiply(weights, rhs, out=rhs)

        Gf = np.ascontiguousarray(Gf)
        toeplitz_ws = None
        apply_w = None
        return PrecomputeState(
            grid=grid,
            weights=weights,
            Gf=Gf,
            rhs=rhs,
            x_center=np.asarray(x_center),
            toeplitz_ws=toeplitz_ws,
            apply_w=apply_w,
        )

    def _apply_A(self, state: PrecomputeState, v: np.ndarray) -> np.ndarray:
        """
        Apply A = D(F^H F)D + lambda I to vector v.
              we dont need D but only d,  D (F^H F)Dv=D (F^H F)Dv=d*t
        """
        # Note: return value may alias internal workspace.
        if self.kernel.dim == 1:
            return self._apply_A_hot_1d(state, v)
        if self.kernel.dim == 2:
            return self._apply_A_hot_2d(state, v)
        weights = state.weights.reshape(-1)
        w = state.apply_w
        if w is None or w.shape != v.shape:
            w = np.empty_like(v, dtype=np.result_type(v.dtype, weights.dtype, np.complex128))
            state.apply_w = w
        np.multiply(weights, v, out=w)
        # t=F^*Fv
        if state.toeplitz_ws is None:
            state.toeplitz_ws = make_toeplitz_workspace(
                state.Gf, state.grid.mtot, self.kernel.dim, dtype=state.Gf.dtype
            )
        if self.kernel.dim == 1:
            t = apply_toeplitz_fft_1d(state.Gf, w, state.grid.mtot, workspace=state.toeplitz_ws)
        else:
            t = apply_toeplitz_fft_nd(
                state.Gf, w, state.grid.mtot, self.kernel.dim, workspace=state.toeplitz_ws
            )
        np.multiply(weights, t, out=t)
        t += self.reg_lambda * v
        return t

    def _apply_A_block(self, state: PrecomputeState, v: np.ndarray) -> np.ndarray:
        """
        Apply A = D(F^H F)D + lambda I to a block of vectors (M, b).
        """
        v = np.asarray(v)
        if v.ndim != 2:
            raise ValueError("v must be a 2D array of shape (M, b).")
        weights = state.weights.reshape(-1)
        if v.shape[0] != weights.size:
            raise ValueError("v has incompatible first dimension.")

        w = state.apply_w_block
        if w is None or w.shape != v.shape:
            w = np.empty_like(v, dtype=np.result_type(v.dtype, weights.dtype, np.complex128))
            state.apply_w_block = w
        np.multiply(weights[:, None], v, out=w)

        ws = state.toeplitz_ws_block
        if ws is None or ws.block_size != v.shape[1]:
            ws = make_toeplitz_block_workspace(
                state.Gf, state.grid.mtot, self.kernel.dim, v.shape[1], dtype=state.Gf.dtype
            )
            state.toeplitz_ws_block = ws

        t = apply_toeplitz_fft_nd_block(
            state.Gf, w, state.grid.mtot, self.kernel.dim, workspace=ws
        )
        np.multiply(weights[:, None], t, out=t)
        t += self.reg_lambda * v
        return t

    def _apply_A_hot_1d(self, state: PrecomputeState, v: np.ndarray) -> np.ndarray:
        weights = state.weights
        w = state.apply_w
        if w is None or w.shape != v.shape:
            w = np.empty_like(v, dtype=np.result_type(v.dtype, weights.dtype, np.complex128))
            state.apply_w = w
        np.multiply(weights, v, out=w)

        ws = state.toeplitz_ws
        if ws is None:
            ws = make_toeplitz_workspace(state.Gf, state.grid.mtot, 1, dtype=state.Gf.dtype)
            state.toeplitz_ws = ws

        pad = ws.pad
        pad.fill(0)
        mtot = state.grid.mtot
        pad[:mtot] = w
        af = _fftn(pad, overwrite_x=_HAS_SCIPY_FFT)
        np.multiply(af, state.Gf, out=af)
        vfft = _ifftn(af, overwrite_x=_HAS_SCIPY_FFT)
        out = ws.out
        np.copyto(out, vfft[mtot - 1 :])
        np.multiply(weights, out, out=out)
        out += self.reg_lambda * v
        return out

    def _apply_A_hot_2d(self, state: PrecomputeState, v: np.ndarray) -> np.ndarray:
        weights = state.weights
        w = state.apply_w
        if w is None or w.shape != v.shape:
            w = np.empty_like(v, dtype=np.result_type(v.dtype, weights.dtype, np.complex128))
            state.apply_w = w
        np.multiply(weights, v, out=w)

        ws = state.toeplitz_ws
        if ws is None:
            ws = make_toeplitz_workspace(state.Gf, state.grid.mtot, 2, dtype=state.Gf.dtype)
            state.toeplitz_ws = ws

        pad = ws.pad
        pad.fill(0)
        mtot = state.grid.mtot
        pad[:mtot, :mtot] = w.reshape(mtot, mtot)
        af = _fftn(pad, overwrite_x=_HAS_SCIPY_FFT)
        np.multiply(af, state.Gf, out=af)
        vfft = _ifftn(af, overwrite_x=_HAS_SCIPY_FFT)
        out = ws.out
        t = vfft[mtot - 1 : 2 * mtot - 1, mtot - 1 : 2 * mtot - 1]
        np.copyto(out, t.reshape(-1))
        np.multiply(weights, out, out=out)
        out += self.reg_lambda * v
        return out

    def _dense_phi(self, x: np.ndarray, state: PrecomputeState) -> np.ndarray:
        x = self._ensure_2d(x)
        tphx = self._scale_from_center(x, state.x_center, state.grid.h)
        if state.multi_index is None:
            m = (state.grid.mtot - 1) // 2
            state.multi_index = generate_multi_index(m, self.kernel.dim)
        J = state.multi_index
        w = state.weights.reshape(-1)
        phase = np.exp(1j * (tphx @ J.T))
        return phase * w[None, :]

    def _dense_A_rhs(
        self, x: np.ndarray, y: np.ndarray, state: PrecomputeState
    ) -> tuple[np.ndarray, np.ndarray]:
        Phi = self._dense_phi(x, state)
        A = Phi.conj().T @ Phi + self.reg_lambda * np.eye(Phi.shape[1], dtype=complex)
        rhs = Phi.conj().T @ y
        return A, rhs

    def solve(
        self,
        x: np.ndarray,
        y: np.ndarray,
        top_q: Optional[int] = None,
        use_richardson: bool = False,
        eta: float = 0.8,
        cg_tol: float = 1e-6,
        cg_maxiter: int = 1000,
        solver_type: Optional[str] = None,
        allow_direct: bool = False,
        eig_method: str = "subspace_iter",
        eig_tol: float = 1e-2,
        eig_maxiter: Optional[int] = 20,
        eig_block_size: Optional[int] = None,
        eig_oversample: int = 2,
    ) -> tuple[np.ndarray, PrecomputeState]:
        """
        Solve for beta with optional EigenPro preconditioning.
        """
        state = self.precompute(x, y)

        solver_kind = solver_type
        if solver_kind is None:
            solver_kind = "richardson" if use_richardson else "pcg"
        if solver_kind == "direct" and not allow_direct:
            raise ValueError("solver_type='direct' requested but allow_direct=False.")
        if solver_kind not in ("pcg", "richardson", "direct"):
            raise ValueError("solver_type must be one of: pcg, richardson, direct.")

        if solver_kind == "direct":
            A_dense, rhs_dense = self._dense_A_rhs(x, y, state)
            beta = np.linalg.solve(A_dense, rhs_dense)
            return beta, state

        precond = None
        if top_q is not None and top_q > 0:
            eigpairs = estimate_top_eigenspace(
                # Only need Av, not A
                lambda v: self._apply_A(state, v),
                size=state.rhs.size,
                top_q=top_q, #actually ouput top_{q+1}, select top_{q+1} as \mu
                method=eig_method,
                tol=eig_tol,
                maxiter=eig_maxiter,
                matvec_block=lambda V: self._apply_A_block(state, V),
                block_size=eig_block_size,
                oversample=eig_oversample,
            )
            if eigpairs.values.size > top_q:
                mu = eigpairs.values[top_q]
            else:
                mu = eigpairs.values[-1]
            precond = build_preconditioner(
                eigpairs.values[:top_q],
                eigpairs.vectors[:, :top_q],
                mu,
            ).apply

        if solver_kind == "richardson":
            x0 = np.zeros_like(state.rhs)
            beta, _, _ = richardson(
                lambda v: self._apply_A(state, v),
                state.rhs,
                x0,
                eta,
                cg_tol,
                cg_maxiter,
                precond=precond,
            )
        else:
            beta, _, _ = pcg(
                lambda v: self._apply_A(state, v),
                state.rhs,
                cg_tol,
                cg_maxiter,
                precond=precond,
            )

        return beta, state

    def predict(self, x: np.ndarray, beta: np.ndarray, state: PrecomputeState) -> np.ndarray:
        """
        Evaluate predictor at x using type-2 NUFFT.
        """
        x = self._ensure_2d(x)
        tphx = self._scale_from_center(x, state.x_center, state.grid.h)
        wbeta = state.weights.reshape(-1) * beta

        if self.kernel.dim == 1:
            yhat = nufft1d2(tphx[:, 0], wbeta, self.nufft_tol, +1)
        elif self.kernel.dim == 2:
            wbeta2 = wbeta.reshape(state.grid.mtot, state.grid.mtot)
            yhat = nufft2d2(tphx[:, 0], tphx[:, 1], wbeta2, self.nufft_tol, +1)
        elif self.kernel.dim == 3:
            wbeta3 = wbeta.reshape(state.grid.mtot, state.grid.mtot, state.grid.mtot)
            yhat = nufft3d2(tphx[:, 0], tphx[:, 1], tphx[:, 2], wbeta3, self.nufft_tol, +1)
        else:
            yhat = nufftnd2(
                tphx,
                wbeta.reshape(-1),
                self.nufft_tol,
                +1,
                n_modes=(state.grid.mtot,) * self.kernel.dim,
            )

        return np.real(yhat)
