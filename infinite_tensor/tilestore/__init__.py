"""Tile storage backends for infinite tensor data.

A TileStore is conceptually two things per registered tensor:
    - a set of windows that have been processed
    - some method of storing those processed windows

The only interface between an InfiniteTensor and its store is:
    - notifying the store when a window is processed (with its output)
    - querying the store for whether a window has been processed
    - reading pixel data back out of the store
    - optional begin_access/end_access hooks so stores with LRU eviction can
      safely defer eviction to the end of a user access
"""

import abc
import warnings
from collections import OrderedDict
from typing import Any, Dict, Optional

import torch


class TileStore(abc.ABC):
    """Abstract base class for tile storage backends."""

    @abc.abstractmethod
    def register_tensor(self, tensor) -> None:
        """Register a tensor instance with the store.

        Called once from ``InfiniteTensor.__init__``. On re-registration with the
        same ``tensor_id``, implementations must validate that the new tensor's
        metadata matches the previously-registered one and raise on mismatch.
        """
        ...

    @abc.abstractmethod
    def clear_tensor(self, tensor_id: str) -> None:
        """Remove all state (registration, windows, cached data) for a tensor."""
        ...

    def clear_cache(self, tensor_id: str) -> None:
        """Drop any regeneratable cached state for a tensor.

        Default implementation is a no-op. Subclasses with regeneratable caches
        (e.g. ``MemoryTileStore``) should override so that subsequent calls to
        :meth:`is_window_processed` return ``False`` for windows that only lived
        in the cache.
        """
        pass

    def begin_access(self, tensor_id: str) -> None:
        """Mark the start of an outer user access (from ``InfiniteTensor.__getitem__``).

        Default implementation is a no-op. Stores that evict based on a size
        limit should track access depth and defer eviction until the outermost
        :meth:`end_access` so windows added during an access can't be evicted
        before :meth:`read_pixels` reads them.
        """
        pass

    def end_access(self, tensor_id: str) -> None:
        """Mark the end of an outer user access. Default implementation is a no-op."""
        pass

    @abc.abstractmethod
    def is_window_processed(self, tensor_id: str, window_index: tuple[int, ...]) -> bool:
        """Return whether a window has been processed for a tensor."""
        ...

    @abc.abstractmethod
    def notify_window_processed(
        self,
        tensor_id: str,
        window_index: tuple[int, ...],
        output: torch.Tensor,
    ) -> None:
        """Record that a window has been processed and hand its output to the store."""
        ...

    @abc.abstractmethod
    def read_pixels(
        self,
        tensor_id: str,
        pixel_slices: tuple[slice, ...],
    ) -> torch.Tensor:
        """Return tensor values for a pixel-space slice.

        All windows intersecting ``pixel_slices`` must already have been passed
        to :meth:`notify_window_processed`; the store is only responsible for
        assembling their outputs into the requested region.
        """
        ...

    @abc.abstractmethod
    def migrate(
        self,
        tensor_id: str,
        old_device: torch.device,
        old_dtype: torch.dtype,
    ) -> None:
        """Migrate stored state for ``tensor_id`` after a ``.to()`` call.

        The caller (``InfiniteTensor.to``) has already updated the registered
        tensor's ``device`` and ``dtype`` to their new values; implementations
        inspect those (via the registered instance) along with the supplied
        ``old_device`` / ``old_dtype`` to decide what storage updates are
        needed.

        Implementations must raise (typically ``ValidationError``) on
        transitions they cannot support, **before** any side effects, so the
        caller can roll back the tensor's declared device/dtype.
        """
        ...

    def get_or_create(
        self,
        tensor_id,
        shape,
        f,
        output_window,
        *,
        args=None,
        args_windows=None,
        dtype=None,
        batch_size=None,
        **kwargs,
    ):
        """Deprecated shim that constructs an :class:`InfiniteTensor`.

        The store already handles idempotent registration (same ``tensor_id``
        + matching metadata is validated by :meth:`register_tensor`), so this
        just forwards to ``InfiniteTensor(...)``. Legacy kwargs are mapped
        onto their modern store-level equivalents:

          - ``cache_limit`` → ``self._cache_size_bytes`` (both stores expose
            this since :class:`PersistentTileStore` was byte-limited).
          - ``tile_size`` → ``self.tile_size`` (persistent stores only;
            :class:`MemoryTileStore` has no tile concept).
          - ``cache_method`` has no modern equivalent; the rewrite dropped
            the direct/indirect split in favor of a single shared LRU.
        """
        warnings.warn(
            "TileStore.get_or_create is deprecated; construct InfiniteTensor "
            "directly with tile_store=store, tensor_id=...",
            DeprecationWarning,
            stacklevel=2,
        )
        kwargs.pop("cache_method", None)
        cache_limit = kwargs.pop("cache_limit", None)
        legacy_tile_size = kwargs.pop("tile_size", None)
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs)}")

        if cache_limit is not None:
            self._cache_size_bytes = cache_limit
        if legacy_tile_size is not None and hasattr(self, "tile_size"):
            self.tile_size = legacy_tile_size

        from infinite_tensor.infinite_tensor import DEFAULT_DTYPE, InfiniteTensor

        return InfiniteTensor(
            shape,
            f,
            output_window,
            args=args,
            args_windows=args_windows,
            dtype=dtype if dtype is not None else DEFAULT_DTYPE,
            tile_store=self,
            tensor_id=tensor_id,
            batch_size=batch_size,
        )

    def clear_direct_caches(self) -> None:
        """Deprecated shim that calls :meth:`clear_cache` for every registered tensor."""
        warnings.warn(
            "TileStore.clear_direct_caches is deprecated; call "
            "InfiniteTensor.clear_cache() or store.clear_cache(tensor_id) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        for tensor_id in list(getattr(self, "_tensor_store", {})):
            self.clear_cache(tensor_id)


def _tensor_bytes(tensor: torch.Tensor) -> int:
    """Return the size in bytes of ``tensor``'s storage (elements only)."""
    return tensor.numel() * tensor.element_size()


class MemoryTileStore(TileStore):
    """In-memory tile store with a single LRU cache shared across all tensors.

    All registered tensors' windows live in one ``OrderedDict`` keyed by
    ``(tensor_id, window_index)``. Eviction is LRU against optional byte and
    window-count limits; window access (via :meth:`read_pixels`) bumps the
    window to MRU.

    Eviction is deferred while any tensor is inside a ``begin_access`` /
    ``end_access`` pair, so a single user operation (including cross-tensor
    processing that spans multiple ``__getitem__`` calls) cannot evict a
    window it is about to read. When the outermost access ends, or when
    ``notify_window_processed`` is called while no access is open, the cache
    is trimmed to its limits.

    Args:
        cache_size_bytes: Byte limit for the cache. ``None`` means unbounded
            on this axis.
        cache_size_windows: Window-count limit for the cache. ``None`` means
            unbounded on this axis.
    """

    def __init__(
        self,
        cache_size_bytes: Optional[int] = None,
        cache_size_windows: Optional[int] = None,
    ):
        super().__init__()
        self._tensor_store: Dict[str, Any] = {}
        self._windows: "OrderedDict[tuple[str, tuple[int, ...]], torch.Tensor]" = OrderedDict()
        self._bytes: int = 0
        self._cache_size_bytes = cache_size_bytes
        self._cache_size_windows = cache_size_windows
        self._access_depth: int = 0

    def register_tensor(self, tensor) -> None:
        """Register a tensor; re-registration requires matching metadata."""
        from infinite_tensor.infinite_tensor import ValidationError

        tensor_id = tensor.uuid
        if tensor_id in self._tensor_store:
            existing_meta = self._tensor_store[tensor_id].to_json()
            new_meta = tensor.to_json()
            if existing_meta != new_meta:
                raise ValidationError(
                    f"Tensor {tensor_id} re-registered with mismatched metadata: "
                    f"existing={existing_meta}, new={new_meta}"
                )
        self._tensor_store[tensor_id] = tensor

    def clear_tensor(self, tensor_id: str) -> None:
        """Forget the tensor and all of its processed windows."""
        self._tensor_store.pop(tensor_id, None)
        self._drop_tensor_windows(tensor_id)

    def clear_cache(self, tensor_id: str) -> None:
        """Drop all stored window outputs for ``tensor_id``; registration is preserved."""
        if tensor_id in self._tensor_store:
            self._drop_tensor_windows(tensor_id)

    def _drop_tensor_windows(self, tensor_id: str) -> None:
        """Remove every cache entry belonging to ``tensor_id``."""
        keys = [key for key in self._windows if key[0] == tensor_id]
        for key in keys:
            output = self._windows.pop(key)
            self._bytes -= _tensor_bytes(output)

    def begin_access(self, tensor_id: str) -> None:
        """Increment the access-depth counter that gates eviction."""
        self._access_depth += 1

    def end_access(self, tensor_id: str) -> None:
        """Decrement the access counter; evict if it returned to zero."""
        self._access_depth -= 1
        assert self._access_depth >= 0, "end_access called without matching begin_access"
        if self._access_depth == 0:
            self._evict()

    def is_window_processed(self, tensor_id: str, window_index: tuple[int, ...]) -> bool:
        return (tensor_id, window_index) in self._windows

    def notify_window_processed(
        self,
        tensor_id: str,
        window_index: tuple[int, ...],
        output: torch.Tensor,
    ) -> None:
        """Store ``output``; evict over-limit entries if no access is open."""
        key = (tensor_id, window_index)
        if key in self._windows:
            self._bytes -= _tensor_bytes(self._windows.pop(key))
        self._windows[key] = output
        self._bytes += _tensor_bytes(output)
        if self._access_depth == 0:
            self._evict()

    def _evict(self) -> None:
        """Evict oldest entries until both configured limits are satisfied."""
        while self._windows and (
            (self._cache_size_bytes is not None and self._bytes > self._cache_size_bytes)
            or (
                self._cache_size_windows is not None
                and len(self._windows) > self._cache_size_windows
            )
        ):
            _, output = self._windows.popitem(last=False)
            self._bytes -= _tensor_bytes(output)

    def _fetch_window(self, tensor_id: str, window_index: tuple[int, ...]) -> torch.Tensor:
        """Return a stored window output, bumping its LRU position."""
        key = (tensor_id, window_index)
        self._windows.move_to_end(key)
        return self._windows[key]

    def read_pixels(
        self,
        tensor_id: str,
        pixel_slices: tuple[slice, ...],
    ) -> torch.Tensor:
        """Assemble the requested pixel region by summing intersecting windows."""
        tensor = self._tensor_store[tensor_id]
        output_window = tensor.output_window

        output_shape = tuple(
            max((s.stop - s.start - 1) // s.step + 1, 0) for s in pixel_slices
        )
        output_tensor = torch.zeros(
            output_shape, dtype=tensor.dtype, device=tensor.device
        )

        for window_index in output_window.intersecting_windows(
            pixel_slices, tensor_shape=tensor.shape
        ):
            window_output = self._fetch_window(tensor_id, window_index)
            window_bounds = output_window.get_bounds(window_index)

            intersected = []
            for req_slice, win_slice in zip(pixel_slices, window_bounds):
                start = max(req_slice.start, win_slice.start)
                stop = min(req_slice.stop, win_slice.stop)
                if req_slice.step > 1 and start < stop:
                    offset = (start - req_slice.start) % req_slice.step
                    if offset:
                        start += req_slice.step - offset
                    if start < stop:
                        trailing = (stop - 1 - req_slice.start) % req_slice.step
                        if trailing:
                            stop -= trailing
                intersected.append(slice(start, stop, req_slice.step))

            if any(s.start >= s.stop for s in intersected):
                continue

            window_local_indices = tuple(
                slice(s.start - w.start, s.stop - w.start, s.step)
                for s, w in zip(intersected, window_bounds)
            )
            output_indices = tuple(
                slice(
                    (s.start - r.start) // r.step,
                    (s.stop - r.start + r.step - 1) // r.step,
                )
                for s, r in zip(intersected, pixel_slices)
            )
            output_tensor[output_indices] += window_output[window_local_indices]

        return output_tensor

    def migrate(
        self,
        tensor_id: str,
        old_device: torch.device,
        old_dtype: torch.dtype,
    ) -> None:
        """Cast/move every cached window for ``tensor_id`` to the tensor's new device/dtype."""
        tensor = self._tensor_store.get(tensor_id)
        if tensor is None:
            return
        new_device = tensor.device
        new_dtype = tensor.dtype
        if new_device == old_device and new_dtype == old_dtype:
            return
        for key, window in list(self._windows.items()):
            if key[0] != tensor_id:
                continue
            self._bytes -= _tensor_bytes(window)
            migrated = window.to(device=new_device, dtype=new_dtype)
            self._windows[key] = migrated
            self._bytes += _tensor_bytes(migrated)


__all__ = ["TileStore", "MemoryTileStore"]
