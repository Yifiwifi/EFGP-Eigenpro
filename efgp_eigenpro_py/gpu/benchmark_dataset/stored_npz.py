"""Zero-copy row-prefix access for uncompressed arrays stored in an NPZ file.

``numpy.savez`` writes each array as an uncompressed NPY member inside a ZIP
container.  Such members can be memory-mapped directly from the NPZ file once
the ZIP and NPY headers have been skipped.  This is useful for Colab scale
sweeps: one 300M-row master artifact can serve 10M, 30M, 100M, and 300M cases
without materializing four duplicate datasets.
"""

from __future__ import annotations

import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


_LOCAL_FILE_HEADER = struct.Struct("<IHHHHHIIIHH")
_LOCAL_FILE_SIGNATURE = 0x04034B50


class StoredNpzError(ValueError):
    """Raised when an NPZ member cannot be memory-mapped safely."""


@dataclass(frozen=True)
class StoredNpyMember:
    """Location and array metadata for one uncompressed NPY member."""

    member_name: str
    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    fortran_order: bool
    data_offset: int
    data_nbytes: int


def _member_name(array_name: str) -> str:
    name = str(array_name)
    return name if name.endswith(".npy") else f"{name}.npy"


def inspect_stored_npy_member(
    npz_path: str | Path,
    array_name: str,
) -> StoredNpyMember:
    """Return mmap metadata for an uncompressed array inside ``npz_path``.

    Compressed ZIP members are rejected because their bytes do not correspond
    directly to the on-disk NPY representation.
    """

    path = Path(npz_path).expanduser().resolve()
    member_name = _member_name(array_name)
    with zipfile.ZipFile(path, "r") as archive:
        try:
            info = archive.getinfo(member_name)
        except KeyError as exc:
            raise StoredNpzError(f"{path.name} has no member {member_name!r}") from exc
        if info.compress_type != zipfile.ZIP_STORED:
            raise StoredNpzError(
                f"{path.name}:{member_name} is compressed and cannot be memory-mapped"
            )

    with path.open("rb") as stream:
        stream.seek(int(info.header_offset))
        raw_header = stream.read(_LOCAL_FILE_HEADER.size)
        if len(raw_header) != _LOCAL_FILE_HEADER.size:
            raise StoredNpzError(f"truncated ZIP local header for {member_name!r}")
        fields = _LOCAL_FILE_HEADER.unpack(raw_header)
        if int(fields[0]) != _LOCAL_FILE_SIGNATURE:
            raise StoredNpzError(f"invalid ZIP local header for {member_name!r}")
        filename_length = int(fields[-2])
        extra_length = int(fields[-1])
        npy_offset = int(info.header_offset) + _LOCAL_FILE_HEADER.size + filename_length + extra_length
        stream.seek(npy_offset)
        try:
            version = np.lib.format.read_magic(stream)
            shape, fortran_order, dtype = np.lib.format._read_array_header(  # type: ignore[attr-defined]
                stream,
                version,
            )
        except Exception as exc:
            raise StoredNpzError(f"invalid NPY header for {member_name!r}") from exc
        data_offset = int(stream.tell())

    dtype = np.dtype(dtype)
    shape = tuple(int(value) for value in shape)
    count = int(np.prod(shape, dtype=np.int64)) if shape else 1
    data_nbytes = int(count * dtype.itemsize)
    member_end = npy_offset + int(info.file_size)
    if data_offset + data_nbytes > member_end:
        raise StoredNpzError(f"array data exceed ZIP member bounds for {member_name!r}")
    return StoredNpyMember(
        member_name=member_name,
        shape=shape,
        dtype=dtype,
        fortran_order=bool(fortran_order),
        data_offset=data_offset,
        data_nbytes=data_nbytes,
    )


def mmap_stored_npz_array(
    npz_path: str | Path,
    array_name: str,
    *,
    mode: str = "r",
) -> np.memmap:
    """Memory-map an uncompressed NPY member directly from its NPZ container."""

    info = inspect_stored_npy_member(npz_path, array_name)
    order = "F" if info.fortran_order else "C"
    return np.memmap(
        Path(npz_path).expanduser().resolve(),
        dtype=info.dtype,
        mode=mode,
        offset=info.data_offset,
        shape=info.shape,
        order=order,
    )


def load_stored_npz_prefix(
    npz_path: str | Path,
    array_name: str,
    n_rows: int | None,
    *,
    dtype: Any | None = None,
) -> np.ndarray:
    """Load a contiguous row prefix without reading the rest of the master array."""

    mapped = mmap_stored_npz_array(npz_path, array_name)
    if mapped.ndim == 0:
        if n_rows not in (None, 0, 1):
            raise ValueError("n_rows is only valid for arrays with a row dimension")
        view = mapped
    else:
        rows = int(mapped.shape[0]) if n_rows in (None, 0) else int(n_rows)
        if rows < 0 or rows > int(mapped.shape[0]):
            raise ValueError(
                f"requested {rows} rows from {array_name!r}, available {mapped.shape[0]}"
            )
        view = mapped[:rows]
    return np.ascontiguousarray(view, dtype=dtype)

