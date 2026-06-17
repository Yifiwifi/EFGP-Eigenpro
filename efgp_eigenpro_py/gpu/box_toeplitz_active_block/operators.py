from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..v1_ops import apply_A_block_v1, apply_A_v1


@dataclass
class GlobalOperatorView:
    backend: Any
    data_ctx: Any
    reg_lambda: float
    op_ctx: Any

    def apply(self, v: Any, out: Any | None = None) -> Any:
        return apply_A_v1(
            self.backend,
            self.data_ctx,
            v,
            float(self.reg_lambda),
            self.op_ctx,
            out=out,
        )

    def apply_block(
        self,
        V: Any,
        *,
        block_cols: int | str = "auto",
        max_workspace_GB: float | None = None,
        out: Any | None = None,
    ) -> Any:
        return apply_A_block_v1(
            self.backend,
            self.data_ctx,
            V,
            float(self.reg_lambda),
            self.op_ctx,
            block_cols=block_cols,
            max_workspace_GB=max_workspace_GB,
            out=out,
        )
