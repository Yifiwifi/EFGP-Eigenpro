"""Build a zero-copy Google Drive catalog for large benchmark ``.npz`` files.

The large benchmark archives in this repository were written with
``numpy.savez`` (ZIP_STORED), not ``numpy.savez_compressed``.  This module uses
the local ZIP headers and NPY headers to memory-map an array directly inside an
NPZ.  Consequently one 300M master can expose the 10M, 30M, 100M, and 300M
nested prefixes without materialising four copies.

The command-line ``prepare`` operation creates either hard links (zero extra
disk blocks on one filesystem) or a manifest-only upload plan.  ``verify``
checks every staged file against its recorded SHA-256 digest.  No network
operation is performed by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
DEFAULT_PREFIX_SIZES = (10_000_000, 30_000_000, 100_000_000, 300_000_000)
REQUIRED_ARRAYS = ("x_train", "y_train")
OPTIONAL_PREFIX_ARRAYS = ("x_test", "y_test")
_LOCAL_FILE_HEADER = struct.Struct("<IHHHHHIIIHH")
_LOCAL_FILE_SIGNATURE = 0x04034B50


@dataclass(frozen=True)
class StoredArray:
    """Description of an NPY array stored without compression inside an NPZ."""

    name: str
    member: str
    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    fortran_order: bool
    data_offset: int
    nbytes: int

    def as_manifest(self) -> dict[str, Any]:
        return {
            "member": self.member,
            "shape": list(self.shape),
            "dtype": self.dtype.str,
            "fortran_order": self.fortran_order,
            "data_offset": self.data_offset,
            "nbytes": self.nbytes,
        }


def sha256_file(path: Path, *, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 without loading a large master into RAM."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _member_payload_offset(path: Path, info: zipfile.ZipInfo) -> int:
    with path.open("rb") as handle:
        handle.seek(info.header_offset)
        payload = handle.read(_LOCAL_FILE_HEADER.size)
    if len(payload) != _LOCAL_FILE_HEADER.size:
        raise ValueError(f"truncated ZIP local header for {info.filename!r}")
    fields = _LOCAL_FILE_HEADER.unpack(payload)
    if fields[0] != _LOCAL_FILE_SIGNATURE:
        raise ValueError(f"invalid ZIP local header for {info.filename!r}")
    filename_bytes = fields[-2]
    extra_bytes = fields[-1]
    return info.header_offset + _LOCAL_FILE_HEADER.size + filename_bytes + extra_bytes


def _read_npy_header(path: Path, payload_offset: int) -> tuple[tuple[int, ...], bool, np.dtype[Any], int]:
    with path.open("rb") as handle:
        handle.seek(payload_offset)
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
        elif version == (3, 0):
            # NumPy's private dispatcher is the only version-3 reader exposed
            # by all supported NumPy releases.  Fall back to the public 2.0
            # reader because the binary layout is otherwise identical.
            reader = getattr(np.lib.format, "_read_array_header", None)
            if reader is None:
                shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
            else:
                shape, fortran_order, dtype = reader(handle, version)
        else:
            raise ValueError(f"unsupported NPY header version {version}")
        data_offset = handle.tell()
    return tuple(int(v) for v in shape), bool(fortran_order), np.dtype(dtype), data_offset


def inspect_stored_npz(path: str | Path) -> dict[str, StoredArray]:
    """Inspect array offsets in a ZIP_STORED NPZ.

    Compressed members deliberately raise an error: a compressed stream cannot
    be memory-mapped and would silently defeat the one-master design.
    """

    archive = Path(path).resolve()
    arrays: dict[str, StoredArray] = {}
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            if info.is_dir() or not info.filename.endswith(".npy"):
                continue
            if info.flag_bits & 0x1:
                raise ValueError(f"encrypted NPZ member is unsupported: {info.filename}")
            if info.compress_type != zipfile.ZIP_STORED:
                raise ValueError(
                    f"{archive.name}:{info.filename} is compressed; create the master "
                    "with numpy.savez (not numpy.savez_compressed)"
                )
            payload_offset = _member_payload_offset(archive, info)
            shape, fortran_order, dtype, data_offset = _read_npy_header(
                archive, payload_offset
            )
            nbytes = int(np.prod(shape, dtype=np.int64)) * int(dtype.itemsize)
            if data_offset + nbytes > payload_offset + info.file_size:
                raise ValueError(f"NPY payload exceeds ZIP member: {info.filename}")
            name = info.filename[:-4]
            if "/" in name or "\\" in name:
                raise ValueError(f"nested NPZ member is unsupported: {info.filename}")
            arrays[name] = StoredArray(
                name=name,
                member=info.filename,
                shape=shape,
                dtype=dtype,
                fortran_order=fortran_order,
                data_offset=data_offset,
                nbytes=nbytes,
            )
    missing = sorted(set(REQUIRED_ARRAYS) - arrays.keys())
    if missing:
        raise ValueError(f"master NPZ is missing required arrays: {missing}")
    return arrays


def open_stored_npz_array(path: str | Path, name: str) -> np.memmap:
    """Open one NPZ member as a read-only memmap without extracting it."""

    archive = Path(path).resolve()
    arrays = inspect_stored_npz(archive)
    if name not in arrays:
        raise KeyError(f"array {name!r} is not present in {archive.name}")
    item = arrays[name]
    order = "F" if item.fortran_order else "C"
    return np.memmap(
        archive,
        mode="r",
        dtype=item.dtype,
        offset=item.data_offset,
        shape=item.shape,
        order=order,
    )


def _validate_training_layout(arrays: Mapping[str, StoredArray]) -> tuple[int, int | None]:
    x_train = arrays["x_train"]
    y_train = arrays["y_train"]
    if len(x_train.shape) != 2 or len(y_train.shape) not in (1, 2):
        raise ValueError("expected x_train rank 2 and y_train rank 1 or 2")
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("x_train and y_train row counts differ")
    n_test: int | None = None
    present_test = [name in arrays for name in OPTIONAL_PREFIX_ARRAYS]
    if any(present_test) and not all(present_test):
        raise ValueError("x_test and y_test must either both be present or both be absent")
    if all(present_test):
        if arrays["x_test"].shape[0] != arrays["y_test"].shape[0]:
            raise ValueError("x_test and y_test row counts differ")
        if len(arrays["x_test"].shape) != 2:
            raise ValueError("expected x_test rank 2")
        if arrays["x_test"].shape[1:] != x_train.shape[1:]:
            raise ValueError("x_train and x_test feature shapes differ")
        n_test = arrays["x_test"].shape[0]
    return x_train.shape[0], n_test


def prefix_plan(
    arrays: Mapping[str, StoredArray], prefix_sizes: Iterable[int]
) -> list[dict[str, Any]]:
    """Describe logical nested prefixes contained in a single master."""

    n_train, n_test = _validate_training_layout(arrays)
    plans: list[dict[str, Any]] = []
    for requested in sorted(set(int(v) for v in prefix_sizes)):
        if requested <= 0:
            raise ValueError("prefix sizes must be positive")
        ready = requested <= n_train
        prefix_test: int | None = None
        if n_test is not None:
            numerator = requested * n_test
            if numerator % n_train:
                raise ValueError(
                    f"prefix {requested} does not preserve the exact test/train ratio "
                    f"{n_test}/{n_train}"
                )
            prefix_test = numerator // n_train
        plans.append(
            {
                "n_train": requested,
                "n_test": prefix_test,
                "status": "ready" if ready else "planned",
                "train_slice": [0, requested],
                "test_slice": None if prefix_test is None else [0, prefix_test],
                "missing_train_rows": max(0, requested - n_train),
            }
        )
    return plans


def open_dataset_prefix(
    path: str | Path,
    n_train: int,
    *,
    include_test: bool = True,
) -> dict[str, np.ndarray]:
    """Return zero-copy prefix views plus small constant arrays.

    The returned training and test arrays are memmap slices.  Other small
    members (for example ``y_mean`` or ``y_center``) are copied because they do
    not have a row-prefix interpretation.
    """

    archive = Path(path).resolve()
    arrays = inspect_stored_npz(archive)
    total_train, total_test = _validate_training_layout(arrays)
    if n_train <= 0 or n_train > total_train:
        raise ValueError(f"n_train must be in [1, {total_train}], got {n_train}")
    result: dict[str, np.ndarray] = {
        "x_train": open_stored_npz_array(archive, "x_train")[:n_train],
        "y_train": open_stored_npz_array(archive, "y_train")[:n_train],
    }
    # Test arrays are either added with their exact prefix below or omitted.
    # They must never fall through to the constants branch, which could load a
    # 75M-row test array when include_test=False.
    row_arrays = set(REQUIRED_ARRAYS) | set(OPTIONAL_PREFIX_ARRAYS)
    if include_test and total_test is not None:
        numerator = n_train * total_test
        if numerator % total_train:
            raise ValueError("requested prefix does not preserve the exact test/train ratio")
        prefix_test = numerator // total_train
        result["x_test"] = open_stored_npz_array(archive, "x_test")[:prefix_test]
        result["y_test"] = open_stored_npz_array(archive, "y_test")[:prefix_test]
    # Preserve extra per-training-row arrays used by the synthetic benchmark.
    for name, item in arrays.items():
        if name in row_arrays:
            continue
        mapped = open_stored_npz_array(archive, name)
        if item.shape and item.shape[0] == total_train:
            result[name] = mapped[:n_train]
        else:
            result[name] = np.asarray(mapped).copy()
    return result


def compare_nested_prefix(
    *,
    larger_npz: str | Path,
    prefix_npz: str | Path,
    chunk_rows: int = 1_000_000,
) -> dict[str, Any]:
    """Prove that a smaller archive is an exact prefix of a larger master.

    Comparison is chunked and memory-mapped, so it is suitable for checking a
    newly generated 300M master against the already frozen 10M artifact.
    """

    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    larger_path = Path(larger_npz).resolve()
    prefix_path = Path(prefix_npz).resolve()
    larger = inspect_stored_npz(larger_path)
    prefix_arrays: dict[str, np.ndarray] | None = None
    try:
        prefix = inspect_stored_npz(prefix_path)
    except ValueError as exc:
        if "is compressed" not in str(exc):
            raise
        # Frozen legacy prefixes may be deflated.  They are much smaller than
        # the master, so decompress only that prefix and still stream the large
        # master through read-only memmaps.
        with np.load(prefix_path, allow_pickle=False) as loaded:
            prefix_arrays = {
                name: np.asarray(loaded[name]) for name in loaded.files
            }
        prefix = {
            name: StoredArray(
                name=name,
                member=f"{name}.npy",
                shape=tuple(int(value) for value in array.shape),
                dtype=np.dtype(array.dtype),
                fortran_order=bool(np.isfortran(array)),
                data_offset=-1,
                nbytes=int(array.nbytes),
            )
            for name, array in prefix_arrays.items()
        }
        missing = sorted(set(REQUIRED_ARRAYS) - prefix.keys())
        if missing:
            raise ValueError(f"prefix NPZ is missing required arrays: {missing}")
    larger_train, larger_test = _validate_training_layout(larger)
    prefix_train, prefix_test = _validate_training_layout(prefix)
    if prefix_train > larger_train:
        raise ValueError("the proposed prefix has more training rows than the master")
    if (prefix_test is None) != (larger_test is None):
        raise ValueError("master and prefix disagree about test arrays")
    if prefix_test is not None and larger_test is not None:
        expected_test_numerator = prefix_train * larger_test
        if expected_test_numerator % larger_train:
            raise ValueError("the prefix size does not preserve the master's test ratio")
        if prefix_test != expected_test_numerator // larger_train:
            raise ValueError("the prefix test row count does not match the master ratio")

    compared: list[str] = []
    for name, smaller_spec in prefix.items():
        if name not in larger:
            raise ValueError(f"master is missing prefix array {name!r}")
        larger_spec = larger[name]
        if smaller_spec.dtype != larger_spec.dtype:
            raise ValueError(f"dtype differs for {name!r}")
        if smaller_spec.shape == larger_spec.shape:
            row_limit = smaller_spec.shape[0] if smaller_spec.shape else 1
        elif (
            smaller_spec.shape
            and larger_spec.shape
            and smaller_spec.shape[1:] == larger_spec.shape[1:]
            and smaller_spec.shape[0] < larger_spec.shape[0]
            and smaller_spec.shape[0] in {prefix_train, prefix_test}
        ):
            row_limit = smaller_spec.shape[0]
        else:
            raise ValueError(f"shape is not prefix-compatible for {name!r}")
        smaller_array = (
            prefix_arrays[name]
            if prefix_arrays is not None
            else open_stored_npz_array(prefix_path, name)
        )
        larger_array = open_stored_npz_array(larger_path, name)
        if smaller_spec.shape:
            for start in range(0, row_limit, chunk_rows):
                stop = min(start + chunk_rows, row_limit)
                if not np.array_equal(smaller_array[start:stop], larger_array[start:stop]):
                    raise ValueError(
                        f"data differs for {name!r} in rows [{start}, {stop})"
                    )
        elif not np.array_equal(smaller_array, larger_array):
            raise ValueError(f"scalar data differs for {name!r}")
        compared.append(name)
    return {
        "larger_npz": str(larger_path),
        "prefix_npz": str(prefix_path),
        "n_train": prefix_train,
        "n_test": prefix_test,
        "arrays_compared": compared,
        "exact_prefix": True,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _stage_hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if os.path.samefile(source, destination):
            return
        raise FileExistsError(
            f"refusing to replace existing staging file {destination}; choose a new "
            "dataset id or remove it after checking its contents"
        )
    try:
        os.link(source, destination)
    except OSError as exc:
        raise OSError(
            f"could not create a hard link from {source} to {destination}. "
            "Both paths must be on one filesystem; use --link-mode manifest-only "
            "when staging on another drive."
        ) from exc


def _parse_prefix_sizes(text: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(text, str):
        values = [int(value.strip()) for value in text.split(",") if value.strip()]
    else:
        values = [int(value) for value in text]
    if not values or any(value <= 0 for value in values):
        raise ValueError("at least one positive prefix size is required")
    return tuple(sorted(set(values)))


def prepare_dataset(
    *,
    source_npz: str | Path,
    output_dir: str | Path,
    dataset_id: str,
    prefix_sizes: Sequence[int] = DEFAULT_PREFIX_SIZES,
    source_metadata: str | Path | None = None,
    link_mode: str = "hardlink",
    bundle_names: Sequence[str] = (),
) -> dict[str, Any]:
    """Add or update one dataset in a Drive upload catalog."""

    source = Path(source_npz).resolve()
    output = Path(output_dir).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not dataset_id or any(character in dataset_id for character in "/\\"):
        raise ValueError("dataset_id must be one non-empty path component")
    if link_mode not in {"hardlink", "manifest-only"}:
        raise ValueError("link_mode must be 'hardlink' or 'manifest-only'")
    requested_bundles = {dataset_id, *(str(name) for name in bundle_names)}
    if any(not name or any(character in name for character in "/\\") for name in requested_bundles):
        raise ValueError("bundle names must be non-empty path-safe labels")

    arrays = inspect_stored_npz(source)
    n_train, n_test = _validate_training_layout(arrays)
    prefixes = prefix_plan(arrays, prefix_sizes)
    relative_npz = Path("data") / dataset_id / source.name
    staged_npz = output / relative_npz
    if link_mode == "hardlink":
        _stage_hardlink(source, staged_npz)

    metadata_path: Path | None
    if source_metadata is None:
        candidate = source.with_suffix(".json")
        metadata_path = candidate if candidate.is_file() else None
    else:
        metadata_path = Path(source_metadata).resolve()
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
    relative_metadata: Path | None = None
    metadata_sha: str | None = None
    metadata_bytes: int | None = None
    if metadata_path is not None:
        relative_metadata = Path("data") / dataset_id / metadata_path.name
        if link_mode == "hardlink":
            _stage_hardlink(metadata_path, output / relative_metadata)
        metadata_sha = sha256_file(metadata_path)
        metadata_bytes = metadata_path.stat().st_size

    dataset_entry: dict[str, Any] = {
        "dataset_id": dataset_id,
        "master": {
            "drive_relative_path": relative_npz.as_posix(),
            "upload_from": str(source),
            "byte_count": source.stat().st_size,
            "sha256": sha256_file(source),
            "storage": "NPZ/ZIP_STORED",
            "n_train": n_train,
            "n_test": n_test,
            "arrays": {name: item.as_manifest() for name, item in arrays.items()},
        },
        "metadata": None
        if metadata_path is None
        else {
            "drive_relative_path": relative_metadata.as_posix(),
            "upload_from": str(metadata_path),
            "byte_count": metadata_bytes,
            "sha256": metadata_sha,
        },
        "logical_prefixes": prefixes,
        "nesting_rule": (
            "Every logical size is the [0:n_train] row prefix of the same master; "
            "the test prefix preserves the master's exact test/train ratio."
        ),
        "staging_mode": link_mode,
        "artifact_names": [
            f"{dataset_id}:master",
            *([] if metadata_path is None else [f"{dataset_id}:metadata"]),
        ],
    }

    artifacts: list[dict[str, Any]] = [
        {
            "name": f"{dataset_id}:master",
            "source_path": str(source),
            "relative_path": relative_npz.as_posix(),
            "size_bytes": source.stat().st_size,
            "sha256": dataset_entry["master"]["sha256"],
            "role": "master_npz",
            "dataset_family": dataset_id,
            "storage": "NPZ/ZIP_STORED",
            "array_schema": dataset_entry["master"]["arrays"],
        }
    ]
    if metadata_path is not None:
        artifacts.append(
            {
                "name": f"{dataset_id}:metadata",
                "source_path": str(metadata_path),
                "relative_path": relative_metadata.as_posix(),
                "size_bytes": metadata_bytes,
                "sha256": metadata_sha,
                "role": "metadata_json",
                "dataset_family": dataset_id,
            }
        )

    catalog_path = output / "drive_manifest.json"
    if catalog_path.is_file():
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        if catalog.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported existing catalog schema: {catalog_path}")
    else:
        catalog = {
            "schema_version": SCHEMA_VERSION,
            "format": "efgp-colab-drive-master-prefix",
            "artifacts": [],
            "bundles": {},
            "datasets": {},
            "upload_note": (
                "Upload each listed master exactly once. Logical experiment sizes "
                "are zero-copy nested prefix views, not separate files."
            ),
        }
    old_names = {
        artifact["name"]
        for artifact in catalog.get("artifacts", [])
        if artifact.get("dataset_family") == dataset_id
    }
    previous_bundles = {
        bundle
        for bundle, names in catalog.get("bundles", {}).items()
        if old_names.intersection(names)
    }
    catalog["artifacts"] = [
        artifact
        for artifact in catalog.get("artifacts", [])
        if artifact.get("dataset_family") != dataset_id
    ]
    catalog["artifacts"].extend(artifacts)
    catalog.setdefault("bundles", {})
    for bundle, names in list(catalog["bundles"].items()):
        catalog["bundles"][bundle] = [name for name in names if name not in old_names]
    new_names = [artifact["name"] for artifact in artifacts]
    for bundle in sorted(requested_bundles | previous_bundles):
        existing = catalog["bundles"].setdefault(bundle, [])
        catalog["bundles"][bundle] = sorted(set(existing) | set(new_names))
    catalog.setdefault("datasets", {})[dataset_id] = dataset_entry
    _atomic_json(catalog_path, catalog)

    dataset_manifest = output / "data" / dataset_id / "dataset_manifest.json"
    _atomic_json(dataset_manifest, dataset_entry)
    _write_checksums(output, catalog)
    return dataset_entry


def add_exact_artifact(
    *,
    source_path: str | Path,
    output_dir: str | Path,
    name: str,
    dataset_family: str,
    role: str = "exact_npz",
    link_mode: str = "hardlink",
    bundle_names: Sequence[str] = (),
) -> dict[str, Any]:
    """Register an exact legacy artifact without imposing prefix semantics.

    This route intentionally accepts compressed NPZ files.  It is used for
    archived paper systems whose sampling/noise definition differs from a
    development scale master and therefore must remain a distinct file.
    """

    source = Path(source_path).resolve()
    output = Path(output_dir).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not name:
        raise ValueError("artifact name must be non-empty")
    if not dataset_family or any(character in dataset_family for character in "/\\"):
        raise ValueError("dataset_family must be one non-empty path component")
    if not role:
        raise ValueError("role must be non-empty")
    if link_mode not in {"hardlink", "manifest-only"}:
        raise ValueError("link_mode must be 'hardlink' or 'manifest-only'")
    requested_bundles = {dataset_family, *(str(value) for value in bundle_names)}
    if any(not value or any(character in value for character in "/\\") for value in requested_bundles):
        raise ValueError("bundle names must be non-empty path-safe labels")

    relative = Path("data") / dataset_family / source.name
    if link_mode == "hardlink":
        _stage_hardlink(source, output / relative)
    artifact = {
        "name": name,
        "source_path": str(source),
        "relative_path": relative.as_posix(),
        "size_bytes": source.stat().st_size,
        "sha256": sha256_file(source),
        "role": role,
        "dataset_family": dataset_family,
    }

    catalog_path = output / "drive_manifest.json"
    if catalog_path.is_file():
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        if catalog.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported existing catalog schema: {catalog_path}")
    else:
        catalog = {
            "schema_version": SCHEMA_VERSION,
            "format": "efgp-colab-drive-master-prefix",
            "artifacts": [],
            "bundles": {},
            "datasets": {},
            "upload_note": (
                "Upload each listed artifact once. Only entries explicitly described "
                "as one-master prefixes may serve multiple logical sizes."
            ),
        }
    old = next(
        (item for item in catalog.get("artifacts", []) if item.get("name") == name),
        None,
    )
    if old is not None and (
        old.get("sha256") != artifact["sha256"]
        or old.get("relative_path") != artifact["relative_path"]
    ):
        raise ValueError(f"artifact name {name!r} already identifies different content")
    catalog["artifacts"] = [
        item for item in catalog.get("artifacts", []) if item.get("name") != name
    ]
    catalog["artifacts"].append(artifact)
    catalog.setdefault("bundles", {})
    for bundle in sorted(requested_bundles):
        catalog["bundles"][bundle] = sorted(
            set(catalog["bundles"].get(bundle, [])) | {name}
        )
    catalog.setdefault("datasets", {})
    _atomic_json(catalog_path, catalog)
    _write_checksums(output, catalog)
    return artifact


def _write_checksums(output: Path, catalog: Mapping[str, Any]) -> None:
    lines = [
        f"{artifact['sha256']}  {artifact['relative_path']}"
        for artifact in sorted(catalog.get("artifacts", []), key=lambda row: row["name"])
    ]
    path = output / "checksums.sha256"
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    os.replace(temporary, path)


def verify_catalog(manifest: str | Path, *, data_root: str | Path | None = None) -> list[str]:
    """Verify staged/Drive data and return the checked relative paths."""

    manifest_path = Path(manifest).resolve()
    catalog = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = Path(data_root).resolve() if data_root is not None else manifest_path.parent
    checked: list[str] = []
    for artifact in catalog.get("artifacts", []):
        relative = artifact["relative_path"]
        path = root / Path(relative)
        if not path.is_file():
            raise FileNotFoundError(f"missing catalog file: {path}")
        actual_bytes = path.stat().st_size
        if actual_bytes != artifact["size_bytes"]:
            raise ValueError(
                f"size mismatch for {relative}: {actual_bytes} != {artifact['size_bytes']}"
            )
        actual_sha = sha256_file(path)
        if actual_sha != artifact["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {relative}")
        if artifact["role"] == "master_npz":
            actual_arrays = inspect_stored_npz(path)
            expected_arrays = artifact["array_schema"]
            for name, expected in expected_arrays.items():
                actual = actual_arrays.get(name)
                if actual is None:
                    raise ValueError(f"missing {name!r} in {relative}")
                if list(actual.shape) != expected["shape"] or actual.dtype.str != expected["dtype"]:
                    raise ValueError(f"array schema mismatch for {relative}:{name}")
        checked.append(relative)
    return checked


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="add one master to a Drive catalog")
    prepare.add_argument("--source-npz", type=Path, required=True)
    prepare.add_argument("--source-metadata", type=Path)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--dataset-id", required=True)
    prepare.add_argument(
        "--prefix-sizes",
        default=",".join(str(value) for value in DEFAULT_PREFIX_SIZES),
        help="comma-separated logical training sizes",
    )
    prepare.add_argument(
        "--link-mode",
        choices=("hardlink", "manifest-only"),
        default="hardlink",
        help="hardlink uses no extra local blocks; manifest-only creates no data entries",
    )
    prepare.add_argument(
        "--bundle",
        action="append",
        default=[],
        help="semantic upload bundle; repeat for multiple bundles",
    )
    exact = subparsers.add_parser(
        "add-artifact",
        help="add an exact legacy artifact without claiming master-prefix equivalence",
    )
    exact.add_argument("--source", type=Path, required=True)
    exact.add_argument("--output-dir", type=Path, required=True)
    exact.add_argument("--name", required=True)
    exact.add_argument("--dataset-family", required=True)
    exact.add_argument("--role", default="exact_npz")
    exact.add_argument(
        "--link-mode",
        choices=("hardlink", "manifest-only"),
        default="hardlink",
    )
    exact.add_argument("--bundle", action="append", default=[])
    verify = subparsers.add_parser("verify", help="verify a staged or uploaded catalog")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument(
        "--data-root",
        type=Path,
        help="directory containing data/; defaults to the manifest directory",
    )
    inspect = subparsers.add_parser("inspect", help="print NPZ offsets and logical prefixes")
    inspect.add_argument("--source-npz", type=Path, required=True)
    inspect.add_argument(
        "--prefix-sizes",
        default=",".join(str(value) for value in DEFAULT_PREFIX_SIZES),
    )
    compare = subparsers.add_parser(
        "compare-prefix", help="prove that a smaller NPZ is a nested prefix"
    )
    compare.add_argument("--larger-npz", type=Path, required=True)
    compare.add_argument("--prefix-npz", type=Path, required=True)
    compare.add_argument("--chunk-rows", type=int, default=1_000_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "prepare":
        entry = prepare_dataset(
            source_npz=args.source_npz,
            source_metadata=args.source_metadata,
            output_dir=args.output_dir,
            dataset_id=args.dataset_id,
            prefix_sizes=_parse_prefix_sizes(args.prefix_sizes),
            link_mode=args.link_mode,
            bundle_names=args.bundle,
        )
        print(json.dumps(entry, indent=2, sort_keys=True))
        return 0
    if args.command == "add-artifact":
        artifact = add_exact_artifact(
            source_path=args.source,
            output_dir=args.output_dir,
            name=args.name,
            dataset_family=args.dataset_family,
            role=args.role,
            link_mode=args.link_mode,
            bundle_names=args.bundle,
        )
        print(json.dumps(artifact, indent=2, sort_keys=True))
        return 0
    if args.command == "verify":
        checked = verify_catalog(args.manifest, data_root=args.data_root)
        print(f"verified {len(checked)} file(s)")
        for path in checked:
            print(path)
        return 0
    if args.command == "compare-prefix":
        result = compare_nested_prefix(
            larger_npz=args.larger_npz,
            prefix_npz=args.prefix_npz,
            chunk_rows=args.chunk_rows,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    arrays = inspect_stored_npz(args.source_npz)
    n_train, n_test = _validate_training_layout(arrays)
    print(
        json.dumps(
            {
                "source_npz": str(args.source_npz.resolve()),
                "n_train": n_train,
                "n_test": n_test,
                "arrays": {name: item.as_manifest() for name, item in arrays.items()},
                "logical_prefixes": prefix_plan(
                    arrays, _parse_prefix_sizes(args.prefix_sizes)
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
