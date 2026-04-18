"""Abstract tile-accumulation store for persistent backends.

A :class:`PersistentTileStore` stores each registered tensor as a grid of
fixed-size tiles. Overlapping window outputs are summed into shared tiles at
``notify_window_processed`` time, so ``read_pixels`` only has to assign tile
data into the requested region.

This module handles everything common to persistent backends:

  - pixel-space / tile-space coordinate math
  - an in-memory LRU cache of decoded tiles
  - window accumulation
  - ``read_pixels`` assembly
  - tracking which windows have been processed

Concrete subclasses (e.g. :class:`HDF5TileStore`) plug in a storage backend by
implementing a handful of primitive I/O methods. They may override
:meth:`PersistentTileStore._validate_registration` for stricter or looser
re-registration rules; the default compares persisted ``metadata`` and
``tile_size`` to the incoming tensor.
"""

import abc
import itertools
from collections import OrderedDict
from typing import Any, Dict, Optional

import torch

from infinite_tensor.tilestore import TileStore


DEFAULT_TILE_SIZE = 512


class PersistentTileStore(TileStore):
    """Abstract persistent tile store that accumulates window outputs into tiles.

    Args:
        tile_size: Per-infinite-dim tile extent. ``int`` applies uniformly; a
            tuple is validated per registered tensor against its infinite-dim
            count.
        tile_cache_size: In-memory LRU size for decoded tiles. ``None`` for
            unbounded.
    """

    def __init__(
        self,
        tile_size: int | tuple[int, ...] = DEFAULT_TILE_SIZE,
        tile_cache_size: Optional[int] = 100,
    ):
        super().__init__()
        self.tile_size = tile_size
        self.tile_cache_size = tile_cache_size
        self._tile_cache: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
        self._tensor_store: Dict[str, Any] = {}
        self._processed_windows_cache: Dict[str, set] = {}
        self._tile_size_cache: Dict[str, tuple[int, ...]] = {}
        self._tensor_access_depth: Dict[str, int] = {}

    # ---- Abstract storage primitives ----

    @abc.abstractmethod
    def _read_tensor_metadata(self, tensor_id: str) -> Optional[dict]:
        """Return ``{"metadata": dict, "tile_size": tuple[int, ...]}`` or ``None``.

        Returns ``None`` when the backend has no record of ``tensor_id``.
        """
        ...

    @abc.abstractmethod
    def _write_tensor_metadata(
        self,
        tensor_id: str,
        metadata: dict,
        tile_size: tuple[int, ...],
    ) -> None:
        """Persist (or overwrite) ``metadata`` and ``tile_size`` for ``tensor_id``."""
        ...

    @abc.abstractmethod
    def _delete_tensor_state(self, tensor_id: str) -> None:
        """Remove all persistent state (metadata, tiles, processed windows) for ``tensor_id``."""
        ...

    @abc.abstractmethod
    def _read_processed_windows(self, tensor_id: str) -> set:
        """Return the on-backend set of processed window indices (empty set if none)."""
        ...

    @abc.abstractmethod
    def _append_processed_window(
        self, tensor_id: str, window_index: tuple[int, ...]
    ) -> None:
        """Append ``window_index`` to the persistent processed-window record."""
        ...

    @abc.abstractmethod
    def _read_tile(
        self, tensor_id: str, tile_index: tuple[int, ...]
    ) -> Optional[torch.Tensor]:
        """Return the stored tile tensor, or ``None`` if absent on the backend."""
        ...

    @abc.abstractmethod
    def _write_tile(
        self, tensor_id: str, tile_index: tuple[int, ...], tile: torch.Tensor
    ) -> None:
        """Persist (or overwrite) ``tile`` for ``tensor_id`` at ``tile_index``."""
        ...

    # ---- Overridable hooks ----

    def _validate_registration(
        self,
        tensor,
        existing: dict,
        effective_tile_size: tuple[int, ...],
    ) -> None:
        """Raise ``ValidationError`` if persisted state disagrees with ``tensor``.

        Called when ``register_tensor`` finds an existing record from
        :meth:`_read_tensor_metadata`. Default compares ``tensor.to_json()`` to
        ``existing['metadata']`` and ``effective_tile_size`` to
        ``existing['tile_size']``. Subclasses may override to skip checks or add
        backend-specific rules.
        """
        from infinite_tensor.infinite_tensor import ValidationError

        new_meta = tensor.to_json()
        if existing["metadata"] != new_meta:
            raise ValidationError(
                f"Tensor {tensor.uuid} re-registered with mismatched metadata: "
                f"existing={existing['metadata']}, new={new_meta}"
            )
        if existing["tile_size"] != effective_tile_size:
            raise ValidationError(
                f"Tensor {tensor.uuid} re-registered with mismatched tile_size: "
                f"existing={existing['tile_size']}, store tile_size={effective_tile_size}"
            )

    def flush(self) -> None:
        """Flush pending backend writes. Default is a no-op."""

    def close(self) -> None:
        """Release backend resources. Default is a no-op."""

    # ---- Access tracking ----

    def begin_access(self, tensor_id: str) -> None:
        """Protect ``tensor_id``'s tiles from tile-cache eviction until ``end_access``.

        Access depth is tracked per tensor. While any tensor has depth > 0,
        its tiles are skipped by :meth:`_evict_tile_cache`, which allows the
        cache to temporarily overrun ``tile_cache_size`` for tiles loaded
        during the ongoing access.
        """
        self._tensor_access_depth[tensor_id] = (
            self._tensor_access_depth.get(tensor_id, 0) + 1
        )

    def end_access(self, tensor_id: str) -> None:
        """Release the tile-cache hold for ``tensor_id`` and trim the cache."""
        depth = self._tensor_access_depth.get(tensor_id, 0) - 1
        assert depth >= 0, "end_access called without matching begin_access"
        if depth == 0:
            self._tensor_access_depth.pop(tensor_id, None)
        else:
            self._tensor_access_depth[tensor_id] = depth
        self._evict_tile_cache()

    # ---- Tile-math helpers ----

    def _effective_tile_size(
        self, tensor_shape: tuple[int | None, ...]
    ) -> tuple[int, ...]:
        """Expand ``self.tile_size`` into a per-infinite-dim tuple for ``tensor_shape``."""
        from infinite_tensor.infinite_tensor import ValidationError

        infinite_dims = sum(1 for d in tensor_shape if d is None)
        if isinstance(self.tile_size, int):
            if self.tile_size <= 0:
                raise ValidationError(f"tile_size must be positive, got {self.tile_size}")
            return (self.tile_size,) * infinite_dims
        if len(self.tile_size) != infinite_dims:
            raise ValidationError(
                f"tile_size tuple length {len(self.tile_size)} != infinite dims {infinite_dims}"
            )
        if any(c <= 0 for c in self.tile_size):
            raise ValidationError(f"All tile sizes must be positive, got {self.tile_size}")
        return tuple(int(x) for x in self.tile_size)

    @staticmethod
    def _tile_shape(
        tensor_shape: tuple[int | None, ...],
        tile_size_tuple: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Full shape of a single tile for this tensor."""
        result: list[int] = []
        infinite_dim_index = 0
        for dim in tensor_shape:
            if dim is None:
                result.append(tile_size_tuple[infinite_dim_index])
                infinite_dim_index += 1
            else:
                result.append(dim)
        return tuple(result)

    @staticmethod
    def _pixel_slices_to_tile_ranges(
        tensor_shape: tuple[int | None, ...],
        tile_size_tuple: tuple[int, ...],
        slices: tuple[slice, ...],
    ) -> tuple[slice, ...]:
        """For each infinite dim, return the tile-space range covering ``slices``."""
        ranges = []
        infinite_dim_index = 0
        for slice_index, pixel_range in enumerate(slices):
            if tensor_shape[slice_index] is None:
                start = pixel_range.start // tile_size_tuple[infinite_dim_index]
                stop = (pixel_range.stop - 1) // tile_size_tuple[infinite_dim_index] + 1
                ranges.append(slice(start, stop))
                infinite_dim_index += 1
        return tuple(ranges)

    @staticmethod
    def _intersect_slices(
        tensor_shape: tuple[int | None, ...],
        tile_size_tuple: tuple[int, ...],
        slices: tuple[slice, ...],
        tile_index: tuple[int, ...],
    ) -> tuple[slice, ...]:
        """Clip ``slices`` to ``tile_index``'s pixel bounds (step-aligned)."""
        out = []
        infinite_dim_index = 0
        for slice_index, pixel_slice in enumerate(slices):
            if tensor_shape[slice_index] is None:
                tile_start = tile_index[infinite_dim_index] * tile_size_tuple[infinite_dim_index]
                tile_end = (tile_index[infinite_dim_index] + 1) * tile_size_tuple[infinite_dim_index]
                start = max(pixel_slice.start, tile_start)
                if pixel_slice.step > 1:
                    offset = (start - pixel_slice.start) % pixel_slice.step
                    if offset:
                        start += pixel_slice.step - offset
                stop = min(pixel_slice.stop, tile_end)
                if pixel_slice.step > 1:
                    offset = (stop - 1 - pixel_slice.start) % pixel_slice.step
                    if offset:
                        stop -= offset
                out.append(slice(start, stop, pixel_slice.step))
                infinite_dim_index += 1
            else:
                out.append(pixel_slice)
        return tuple(out)

    @staticmethod
    def _translate_slices(
        tensor_shape: tuple[int | None, ...],
        tile_size_tuple: tuple[int, ...],
        slices: tuple[slice, ...],
        tile_index: tuple[int, ...],
    ) -> tuple[slice, ...]:
        """Subtract tile origin from ``slices`` so they index into tile-local space."""
        out = []
        infinite_dim_index = 0
        for slice_index, pixel_slice in enumerate(slices):
            if tensor_shape[slice_index] is None:
                offset = tile_index[infinite_dim_index] * tile_size_tuple[infinite_dim_index]
                out.append(
                    slice(
                        pixel_slice.start - offset,
                        pixel_slice.stop - offset,
                        pixel_slice.step,
                    )
                )
                infinite_dim_index += 1
            else:
                out.append(pixel_slice)
        return tuple(out)

    # ---- Tile LRU cache ----

    def _cache_tile(
        self, tensor_id: str, tile_index: tuple[int, ...], tile: torch.Tensor
    ) -> None:
        """Insert or refresh a tile in the LRU cache."""
        key = (tensor_id, tile_index)
        if key in self._tile_cache:
            self._tile_cache[key] = tile
            self._tile_cache.move_to_end(key)
            return
        self._tile_cache[key] = tile
        self._evict_tile_cache()

    def _evict_tile_cache(self) -> None:
        """Trim the tile cache to ``tile_cache_size``, skipping protected tensors.

        Tiles whose tensor has a positive ``begin_access``/``end_access`` depth
        are held in place; if every oldest entry belongs to a protected tensor
        the cache is allowed to exceed its limit until the protecting access
        ends and calls this method again.
        """
        if self.tile_cache_size is None:
            return
        overflow = len(self._tile_cache) - self.tile_cache_size
        if overflow <= 0:
            return
        victims: list[tuple] = []
        for key in self._tile_cache:
            if len(victims) >= overflow:
                break
            tile_tensor_id, _ = key
            if self._tensor_access_depth.get(tile_tensor_id, 0) == 0:
                victims.append(key)
        for key in victims:
            del self._tile_cache[key]

    def _get_cached_tile(
        self, tensor_id: str, tile_index: tuple[int, ...]
    ) -> Optional[torch.Tensor]:
        """Return a cached tile and mark it most-recently-used, or ``None``."""
        key = (tensor_id, tile_index)
        if key in self._tile_cache:
            self._tile_cache.move_to_end(key)
            return self._tile_cache[key]
        return None

    def _materialize_tile(self, tensor_id: str, tile: torch.Tensor) -> torch.Tensor:
        """Return ``tile`` on the registered tensor's device (no-op if already matching)."""
        tensor = self._tensor_store.get(tensor_id)
        if tensor is None:
            return tile
        if tile.device == tensor.device:
            return tile
        return tile.to(tensor.device)

    def _load_tile(
        self, tensor_id: str, tile_index: tuple[int, ...]
    ) -> Optional[torch.Tensor]:
        """Return the tile (cache first, then backend), or ``None`` if absent."""
        cached = self._get_cached_tile(tensor_id, tile_index)
        if cached is not None:
            return cached
        tile = self._read_tile(tensor_id, tile_index)
        if tile is not None:
            tile = self._materialize_tile(tensor_id, tile)
            self._cache_tile(tensor_id, tile_index, tile)
        return tile

    def _store_tile(
        self, tensor_id: str, tile_index: tuple[int, ...], tile: torch.Tensor
    ) -> None:
        """Refresh the cache and persist the tile via the backend."""
        self._cache_tile(tensor_id, tile_index, tile)
        self._write_tile(tensor_id, tile_index, tile)

    # ---- Processed-windows cache ----

    def _load_processed_windows(self, tensor_id: str) -> set:
        """Return the set of processed window indices, loading from backend on first access."""
        if tensor_id in self._processed_windows_cache:
            return self._processed_windows_cache[tensor_id]
        result = self._read_processed_windows(tensor_id)
        self._processed_windows_cache[tensor_id] = result
        return result

    # ---- TileStore interface ----

    def register_tensor(self, tensor) -> None:
        """Register a tensor; on re-registration, validate against persisted metadata."""
        tensor_id = tensor.uuid
        new_meta = tensor.to_json()
        effective_tile_size = self._effective_tile_size(tensor.shape)

        existing = self._read_tensor_metadata(tensor_id)
        if existing is not None:
            self._validate_registration(tensor, existing, effective_tile_size)
            self._tensor_store[tensor_id] = tensor
            self._tile_size_cache[tensor_id] = tuple(existing["tile_size"])
            return

        self._write_tensor_metadata(tensor_id, new_meta, effective_tile_size)
        self._tensor_store[tensor_id] = tensor
        self._tile_size_cache[tensor_id] = effective_tile_size
        self._processed_windows_cache[tensor_id] = set()
        self.flush()

    def clear_tensor(self, tensor_id: str) -> None:
        """Delete all backend and in-memory state for ``tensor_id``."""
        for key in [k for k in self._tile_cache if k[0] == tensor_id]:
            del self._tile_cache[key]
        self._tensor_store.pop(tensor_id, None)
        self._processed_windows_cache.pop(tensor_id, None)
        self._tile_size_cache.pop(tensor_id, None)
        self._tensor_access_depth.pop(tensor_id, None)
        self._delete_tensor_state(tensor_id)
        self.flush()

    def is_window_processed(
        self, tensor_id: str, window_index: tuple[int, ...]
    ) -> bool:
        return window_index in self._load_processed_windows(tensor_id)

    def notify_window_processed(
        self,
        tensor_id: str,
        window_index: tuple[int, ...],
        output: torch.Tensor,
    ) -> None:
        """Accumulate ``output`` into the covered tiles and mark the window processed."""
        tensor = self._tensor_store[tensor_id]
        tensor_shape = tensor.shape
        tile_size_tuple = self._tile_size_cache[tensor_id]
        raw_bounds = tensor.output_window.get_bounds(window_index)
        pixel_bounds = tuple(
            slice(b.start, b.stop, b.step if b.step is not None else 1)
            for b in raw_bounds
        )

        tile_range_slices = self._pixel_slices_to_tile_ranges(
            tensor_shape, tile_size_tuple, pixel_bounds
        )
        tile_ranges = [range(s.start, s.stop) for s in tile_range_slices]
        tile_shape = self._tile_shape(tensor_shape, tile_size_tuple)

        for tile_index in itertools.product(*tile_ranges):
            tile = self._load_tile(tensor_id, tile_index)
            if tile is None:
                tile = torch.zeros(
                    tile_shape, dtype=tensor.dtype, device=tensor.device
                )

            intersected = self._intersect_slices(
                tensor_shape, tile_size_tuple, pixel_bounds, tile_index
            )
            tile_local = self._translate_slices(
                tensor_shape, tile_size_tuple, intersected, tile_index
            )
            value_local = tuple(
                slice(
                    (s.start - b.start) // s.step,
                    (s.stop - b.start - 1) // s.step + 1,
                )
                for s, b in zip(intersected, pixel_bounds)
            )

            tile[tile_local] += output[value_local]
            self._store_tile(tensor_id, tile_index, tile)

        processed = self._load_processed_windows(tensor_id)
        assert window_index not in processed, (
            f"Window {window_index} already processed for tensor {tensor_id}"
        )
        processed.add(window_index)
        self._append_processed_window(tensor_id, window_index)
        self.flush()

    def read_pixels(
        self,
        tensor_id: str,
        pixel_slices: tuple[slice, ...],
    ) -> torch.Tensor:
        """Assemble the pixel region by assigning from accumulated tiles."""
        from infinite_tensor.infinite_tensor import TileAccessError, TILE_DELETED_ERROR_MSG

        tensor = self._tensor_store[tensor_id]
        tensor_shape = tensor.shape
        tile_size_tuple = self._tile_size_cache[tensor_id]

        output_shape = tuple(
            max((s.stop - s.start - 1) // s.step + 1, 0) for s in pixel_slices
        )
        output_tensor = torch.empty(
            output_shape, dtype=tensor.dtype, device=tensor.device
        )

        tile_range_slices = self._pixel_slices_to_tile_ranges(
            tensor_shape, tile_size_tuple, pixel_slices
        )
        tile_ranges = [range(s.start, s.stop) for s in tile_range_slices]

        for tile_index in itertools.product(*tile_ranges):
            tile = self._load_tile(tensor_id, tile_index)
            if tile is None:
                raise TileAccessError(TILE_DELETED_ERROR_MSG)
            intersected = self._intersect_slices(
                tensor_shape, tile_size_tuple, pixel_slices, tile_index
            )
            tile_local = self._translate_slices(
                tensor_shape, tile_size_tuple, intersected, tile_index
            )
            output_indices = tuple(
                slice(
                    (s.start - r.start) // r.step,
                    (s.stop - r.start + r.step - 1) // r.step,
                )
                for s, r in zip(intersected, pixel_slices)
            )
            output_tensor[output_indices] = tile[tile_local]

        return output_tensor

    def migrate(
        self,
        tensor_id: str,
        old_device: torch.device,
        old_dtype: torch.dtype,
    ) -> None:
        """Migrate ``tensor_id`` after a ``.to()``; persistent stores forbid dtype changes.

        Raises :class:`ValidationError` (before any side effects) if the new
        dtype differs from ``old_dtype``. Otherwise drops this tensor's cached
        tiles so they are re-materialized on the new device next read, and
        rewrites the persisted metadata so the on-disk ``to_json`` matches the
        new declared device.
        """
        from infinite_tensor.infinite_tensor import ValidationError

        tensor = self._tensor_store.get(tensor_id)
        if tensor is None:
            return
        if tensor.dtype != old_dtype:
            raise ValidationError(
                f"Persistent stores do not support dtype changes for tensor {tensor_id}; "
                f"clear the tensor and re-create it to change dtype"
            )
        if tensor.device != old_device:
            for key in [k for k in self._tile_cache if k[0] == tensor_id]:
                del self._tile_cache[key]
        self._write_tensor_metadata(
            tensor_id, tensor.to_json(), self._tile_size_cache[tensor_id]
        )
        self.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
