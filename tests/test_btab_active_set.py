import numpy as np

from efgp_eigenpro_py.gpu.box_toeplitz_active_block.active_set import (
    select_active_indices,
)


def test_tau_selection_uses_strict_threshold() -> None:
    rho = np.array([0.5, 1.0, 1.5])

    selected = select_active_indices(rho, mode="tau", tau=1.0)

    np.testing.assert_array_equal(selected, np.array([2], dtype=np.int64))
