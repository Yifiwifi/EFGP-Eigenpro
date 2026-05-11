from __future__ import annotations

"""Backward-compatible size-sweep entry point for contiguous USGS 3DEP LiDAR preprocessing.

This wrapper keeps the old file role but delegates to preprocess_usgs_3dep_lidar_contiguous.py,
which now supports both single-run preprocessing and N_train sweeps.
"""

import sys

from preprocess_usgs_3dep_lidar_contiguous import DEFAULT_N_TRAIN_LIST, main as _merged_main


if __name__ == "__main__":
    # If the user runs the legacy size-sweep script without an explicit size list,
    # enable the default N_train sweep.  If they already passed --n-train-list or
    # --size-list, leave argv unchanged.
    has_size_arg = any(arg.startswith("--n-train-list") or arg.startswith("--size-list") for arg in sys.argv[1:])
    if not has_size_arg:
        sys.argv.extend(["--n-train-list", ",".join(str(v) for v in DEFAULT_N_TRAIN_LIST)])
    _merged_main()
