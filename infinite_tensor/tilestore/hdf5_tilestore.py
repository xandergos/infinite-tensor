"""HDF5 backend for :class:`PersistentTileStore`.

Stores each registered tensor as an HDF5 group
``/tensors/<tensor_id>/`` containing:

  - ``metadata`` (group with ``attrs['meta']`` = JSON of ``tensor.to_json()``
    and ``attrs['tile_size']`` = per-infinite-dim tile extent).
  - ``processed_windows`` (vlen string dataset of encoded window indices).
  - ``tiles/`` (group of per-tile-index compressed datasets).

Write-once per window: once ``notify_window_processed`` records a window it
cannot be un-processed. ``clear_tensor`` is the only way to wipe per-tensor
state.

Device handling is transparent: tiles are always serialized to HDF5 as CPU
numpy arrays, and are transferred onto the registered tensor's declared
device when loaded (see :meth:`PersistentTileStore._materialize_tile`). This
backend therefore supports any ``torch.device`` on the owning
:class:`InfiniteTensor`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch

from infinite_tensor.tilestore.persistent import (
    DEFAULT_CACHE_SIZE_BYTES,
    DEFAULT_TILE_SIZE,
    PersistentTileStore,
)

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    h5py = None


class HDF5TileStore(PersistentTileStore):
    """HDF5-backed :class:`PersistentTileStore`.

    Re-registration uses the base class validation (``tensor.to_json()`` and
    effective ``tile_size`` must match what is already on disk).

    Args:
        filepath: Path to the HDF5 file (created if missing).
        mode: ``'r'`` read-only, ``'a'`` append (default), or ``'w'`` to truncate.
        compression: Compression algorithm for tile datasets.
        compression_opts: Compression level (0-9 for gzip).
        tile_size: Per-infinite-dim tile extent. ``int`` applies uniformly; a
            tuple is validated per registered tensor against its infinite-dim
            count. Defaults to 512.
        cache_size_bytes: Byte limit for the in-memory tile LRU cache. ``None``
            means unbounded on this axis. Defaults to 100 MiB.
        cache_size_tiles: Tile-count limit for the in-memory tile LRU cache.
            ``None`` means unbounded on this axis.

    Deprecated kwargs (accepted via ``**kwargs`` for backward compatibility):
        tile_cache_size: Old name for ``cache_size_tiles``.

    Raises:
        ImportError: If h5py is not installed.
    """

    def __init__(
        self,
        filepath: str | Path,
        mode: str = "a",
        compression: str | None = "gzip",
        compression_opts: int | None = 4,
        tile_size: int | tuple[int, ...] = DEFAULT_TILE_SIZE,
        cache_size_bytes: Optional[int] = DEFAULT_CACHE_SIZE_BYTES,
        cache_size_tiles: Optional[int] = None,
        **kwargs,
    ):
        if not HAS_H5PY:
            raise ImportError(
                "h5py is required for HDF5TileStore. "
                "Install it with: pip install infinite-tensor[hdf5]"
            )
        super().__init__(
            tile_size=tile_size,
            cache_size_bytes=cache_size_bytes,
            cache_size_tiles=cache_size_tiles,
            **kwargs,
        )
        self.filepath = Path(filepath)
        self.compression = compression
        self.compression_opts = compression_opts

        if mode == "w":
            with h5py.File(self.filepath, "w") as file_handle:
                file_handle.create_group("tensors")
            self.mode = "a"
        elif mode == "r":
            self.mode = "r"
        else:
            self.mode = "a"

        self._file: Optional["h5py.File"] = None

        if self.mode != "r":
            self._ensure_file_exists()
        self._open_file()

    # ---- File handle management ----

    def _ensure_file_exists(self) -> None:
        """Create the file and the root ``tensors`` group if they don't exist."""
        with h5py.File(self.filepath, "a") as file_handle:
            if "tensors" not in file_handle:
                file_handle.create_group("tensors")

    def _open_file(self) -> None:
        """Open the persistent HDF5 handle if it isn't already open."""
        if self._file is None:
            self._file = h5py.File(self.filepath, self.mode)

    def _get_file(self):
        """Return the active HDF5 file handle."""
        if self._file is None:
            raise RuntimeError("HDF5TileStore file handle is closed")
        return self._file

    def _close_file(self) -> None:
        """Close the underlying HDF5 file handle."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def _close_backend(self) -> None:
        """Close the persistent HDF5 file handle."""
        self._close_file()

    def _flush_backend(self) -> None:
        """Flush HDF5 buffers to disk if the file is open."""
        if self._file is not None:
            self._file.flush()

    def __enter__(self):
        self._open_file()
        return self

    # ---- Index encoding ----

    @staticmethod
    def _encode_index(index: tuple[int, ...]) -> str:
        """Encode an integer tuple as ``"a_b_c"`` for use as an HDF5 key."""
        return "_".join(map(str, index))

    @staticmethod
    def _decode_index(encoded: str) -> tuple[int, ...]:
        """Inverse of :meth:`_encode_index`."""
        return tuple(map(int, encoded.split("_")))

    def _get_tensor_group(self, tensor_id: str, create: bool = False):
        """Return the HDF5 group for a tensor, optionally creating it."""
        file_handle = self._get_file()
        path = f"tensors/{tensor_id}"
        if path in file_handle:
            return file_handle[path]
        if create:
            group = file_handle.create_group(path)
            group.create_group("tiles")
            group.create_group("metadata")
            return group
        return None

    # ---- Storage primitives ----

    def _read_tensor_metadata(self, tensor_id: str) -> Optional[dict]:
        from infinite_tensor.infinite_tensor import ValidationError

        tensor_group = self._get_tensor_group(tensor_id)
        if tensor_group is None:
            return None
        metadata_group = tensor_group["metadata"]
        stored_meta_json = metadata_group.attrs.get("meta")
        if stored_meta_json is None:
            raise ValidationError(f"Tensor {tensor_id} has no metadata on disk")
        return {
            "metadata": json.loads(stored_meta_json),
            "tile_size": tuple(int(x) for x in metadata_group.attrs["tile_size"]),
        }

    def _write_tensor_metadata(
        self,
        tensor_id: str,
        metadata: dict,
        tile_size: tuple[int, ...],
    ) -> None:
        tensor_group = self._get_tensor_group(tensor_id, create=True)
        metadata_group = tensor_group["metadata"]
        metadata_group.attrs["meta"] = json.dumps(metadata)
        metadata_group.attrs["tile_size"] = list(tile_size)
        if "processed_windows" not in tensor_group:
            vlen_str_dtype = h5py.special_dtype(vlen=str)
            tensor_group.create_dataset(
                "processed_windows", shape=(0,), maxshape=(None,), dtype=vlen_str_dtype
            )

    def _delete_tensor_state(self, tensor_id: str) -> None:
        file_handle = self._get_file()
        path = f"tensors/{tensor_id}"
        if path in file_handle:
            del file_handle[path]

    def _read_processed_windows(self, tensor_id: str) -> set:
        tensor_group = self._get_tensor_group(tensor_id)
        if tensor_group is None or "processed_windows" not in tensor_group:
            return set()
        dataset = tensor_group["processed_windows"]
        raw_entries = [w.decode() if isinstance(w, bytes) else w for w in dataset[:]]
        return {self._decode_index(w) for w in raw_entries}

    def _append_processed_window(
        self, tensor_id: str, window_index: tuple[int, ...]
    ) -> None:
        tensor_group = self._get_tensor_group(tensor_id, create=True)
        if "processed_windows" not in tensor_group:
            vlen_str_dtype = h5py.special_dtype(vlen=str)
            tensor_group.create_dataset(
                "processed_windows", shape=(0,), maxshape=(None,), dtype=vlen_str_dtype
            )
        dataset = tensor_group["processed_windows"]
        current_length = dataset.shape[0]
        dataset.resize((current_length + 1,))
        dataset[current_length] = self._encode_index(window_index)

    def _read_tile(
        self, tensor_id: str, tile_index: tuple[int, ...]
    ) -> Optional[tuple[torch.Tensor, set[tuple[int, ...]]]]:
        tensor_group = self._get_tensor_group(tensor_id)
        if tensor_group is None:
            return None
        tiles_group = tensor_group["tiles"]
        encoded = self._encode_index(tile_index)
        if encoded not in tiles_group:
            return None
        dataset = tiles_group[encoded]
        tile_tensor = torch.from_numpy(dataset[:])
        raw_contributions = dataset.attrs.get("contributions", [])
        contributions = {
            self._decode_index(entry.decode() if isinstance(entry, bytes) else entry)
            for entry in raw_contributions
        }
        return tile_tensor, contributions

    def _write_tile(
        self,
        tensor_id: str,
        tile_index: tuple[int, ...],
        tile: torch.Tensor,
        contributions: set[tuple[int, ...]],
    ) -> None:
        tensor_group = self._get_tensor_group(tensor_id, create=True)
        tiles_group = tensor_group["tiles"]
        encoded = self._encode_index(tile_index)
        tile_numpy = tile.cpu().numpy()
        if encoded in tiles_group:
            del tiles_group[encoded]
        dataset = tiles_group.create_dataset(
            encoded,
            data=tile_numpy,
            compression=self.compression,
            compression_opts=self.compression_opts,
        )
        encoded_contributions = [
            self._encode_index(window_index) for window_index in contributions
        ]
        vlen_str_dtype = h5py.special_dtype(vlen=str)
        dataset.attrs.create(
            "contributions", encoded_contributions, dtype=vlen_str_dtype
        )

