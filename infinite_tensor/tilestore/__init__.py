"""Tile storage backends for infinite tensor data.

This module provides the storage abstraction layer for InfinityTensor tiles.
Tiles are sections of tensor data that fit in memory, and the tile store manages
their lifecycle, retrieval, and cleanup.

The abstract TileStore interface allows for different storage strategies:
- MemoryTileStore: Fast in-memory storage (default)
- HDF5TileStore: Persistent HDF5-backed storage for large datasets
"""

import abc
import uuid
import numpy as np
from collections import OrderedDict
from typing import Dict, Any, Optional, Tuple, Iterable, Set

class TileStore(abc.ABC):
    """Abstract base class for tile storage backends.
    
    TileStore defines the interface for storing, retrieving, and managing
    InfinityTensorTile objects. Different implementations can provide
    various storage strategies (memory, disk, distributed, etc.).
    
    The tile store tracks per-tensor window processing state to prevent
    duplicate computation and enable efficient memory management.
    """
    def register_tensor_meta(self, tensor_id: str, meta: dict) -> None:
        """Register a new tensor and its metadata in the store."""
        raise NotImplementedError

    def get_tensor_meta(self, tensor_id: str) -> dict:
        """Fetch metadata dict for a tensor."""
        raise NotImplementedError
    
    def clear_tensor(self, tensor_id: str) -> None:
        """Remove all state for a tensor (tiles, windows, metadata)."""
        raise NotImplementedError

    # Tile operations scoped to a tensor
    def get_tile_for(self, tensor_id: str, tile_index: tuple[int, ...]):
        raise NotImplementedError

    def set_tile_for(self, tensor_id: str, tile_index: tuple[int, ...], value) -> None:
        raise NotImplementedError

    def delete_tile_for(self, tensor_id: str, tile_index: tuple[int, ...]) -> None:
        raise NotImplementedError

    def iter_tile_keys_for(self, tensor_id: str) -> Iterable[tuple[int, ...]]:
        raise NotImplementedError

    # Window cache operations for direct caching
    def cache_window_for(self, tensor_id: str, window_index: tuple[int, ...], output) -> None:
        """Cache a window output without eviction."""
        raise NotImplementedError

    def evict_cache_for(self, tensor_id: str, cache_limit: Optional[int]) -> None:
        """Evict oldest cache entries until under the byte limit."""
        raise NotImplementedError

    def promote_windows_for(self, tensor_id: str, window_indices: list[tuple[int, ...]]) -> None:
        """Move specified windows to end of cache (most recently used) in order."""
        raise NotImplementedError

    def get_cached_window_for(self, tensor_id: str, window_index: tuple[int, ...]):
        """Get a cached window output, marking as recently used. Returns None if not cached."""
        raise NotImplementedError

    def is_window_cached_for(self, tensor_id: str, window_index: tuple[int, ...]) -> bool:
        """Check if a window is in the cache."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_or_create(self,
                      tensor_id,
                      shape: tuple[int|None, ...],
                      f,
                      output_window,
                      args: tuple = None,
                      args_windows = None,
                      tile_size: int | tuple[int, ...] = 512,
                      dtype=None,
                      batch_size: int | None = None,
                      cache_method: str = 'indirect',
                      cache_limit: int | None = 10 * 1024 * 1024):
        """Create or return an InfiniteTensor bound to this store.
        
        Args:
            cache_method: Caching strategy - 'indirect' (default) stores tiles, 
                         'direct' caches window outputs directly.
            cache_limit: Maximum cache size in bytes for direct caching (default: 10MB).
                        None for unlimited cache.
        """
        raise NotImplementedError
    
    def get_tensor(self, tensor_id: str):
        """Get a tensor by it's ID. Raises a KeyError if the tensor is not found."""
        raise NotImplementedError

class MemoryTileStore(TileStore):
    """In-memory tile storage implementation.
    
    This is the default tile store that keeps all tiles in RAM using
    Python dictionaries. Provides fast access but is limited by available
    memory. Suitable for moderate-sized datasets or when disk I/O should
    be avoided.
    """
    def __init__(self):
        """Initialize an empty in-memory tile store."""
        super().__init__()
        self._tile_store: Dict[tuple, Any] = {}
        self._tensor_store: Dict[str, Any] = {}
        self._tensor_meta: Dict[str, Any] = {}
        self._processed_windows_by_tensor: Dict[str, Set[Tuple[int, ...]]] = {}
        # Window cache for direct caching (per-tensor LRU cache)
        self._window_cache: Dict[str, OrderedDict] = {}
        self._window_cache_size: Dict[str, int] = {}

    def register_tensor_meta(self, tensor_id: str, meta: dict) -> None:
        self._tensor_meta[tensor_id] = meta
        self._processed_windows_by_tensor[tensor_id] = set()
        self._window_cache[tensor_id] = OrderedDict()
        self._window_cache_size[tensor_id] = 0

    def get_tensor_meta(self, tensor_id: str) -> dict:
        return self._tensor_meta[tensor_id]
    
    def clear_tensor(self, tensor_id: str) -> None:
        # Remove tiles
        for tile_idx in list(self.iter_tile_keys_for(tensor_id)):
            self.delete_tile_for(tensor_id, tile_idx)
        # Remove windows
        if tensor_id in self._processed_windows_by_tensor:
            del self._processed_windows_by_tensor[tensor_id]
        # Remove window cache
        if tensor_id in self._window_cache:
            del self._window_cache[tensor_id]
        if tensor_id in self._window_cache_size:
            del self._window_cache_size[tensor_id]
        # Remove metadata
        if tensor_id in self._tensor_meta:
            del self._tensor_meta[tensor_id]
        # Remove tensor instance
        if tensor_id in self._tensor_store:
            del self._tensor_store[tensor_id]

    def get_tile_for(self, tensor_id: str, tile_index: tuple[int, ...]):
        return self._tile_store.get((tensor_id, tile_index))

    def set_tile_for(self, tensor_id: str, tile_index: tuple[int, ...], value) -> None:
        self._tile_store[(tensor_id, tile_index)] = value

    def delete_tile_for(self, tensor_id: str, tile_index: tuple[int, ...]) -> None:
        key = (tensor_id, tile_index)
        if key in self._tile_store:
            del self._tile_store[key]

    def iter_tile_keys_for(self, tensor_id: str) -> Iterable[tuple[int, ...]]:
        for key in self._tile_store.keys():
            if isinstance(key, tuple) and len(key) == 2 and key[0] == tensor_id:
                yield key[1]

    def is_window_processed_for(self, tensor_id: str, window_index: tuple[int, ...]) -> bool:
        return window_index in self._processed_windows_by_tensor.get(tensor_id, set())

    def mark_window_processed_for(self, tensor_id: str, window_index: tuple[int, ...]) -> None:
        self._processed_windows_by_tensor.setdefault(tensor_id, set()).add(window_index)

    def _tensor_size_bytes(self, t) -> int:
        """Calculate the size of a tensor in bytes."""
        return t.numel() * t.element_size()

    def cache_window_for(self, tensor_id: str, window_index: tuple[int, ...], output) -> None:
        """Cache a window output without eviction."""
        cache = self._window_cache.get(tensor_id)
        if cache is None:
            self._window_cache[tensor_id] = OrderedDict()
            self._window_cache_size[tensor_id] = 0
            cache = self._window_cache[tensor_id]
        
        if window_index in cache:
            cache.move_to_end(window_index)
            return
        
        output_bytes = self._tensor_size_bytes(output)
        cache[window_index] = output
        self._window_cache_size[tensor_id] += output_bytes

    def evict_cache_for(self, tensor_id: str, cache_limit: Optional[int]) -> None:
        """Evict oldest cache entries until under the byte limit."""
        if cache_limit is None:
            return
        cache = self._window_cache.get(tensor_id)
        if cache is None:
            return
        while self._window_cache_size[tensor_id] > cache_limit and len(cache) > 1:
            _, evicted = cache.popitem(last=False)
            self._window_cache_size[tensor_id] -= self._tensor_size_bytes(evicted)

    def promote_windows_for(self, tensor_id: str, window_indices: list[tuple[int, ...]]) -> None:
        """Move specified windows to end of cache (most recently used) in order."""
        cache = self._window_cache.get(tensor_id)
        if cache is None:
            return
        for window_index in window_indices:
            if window_index in cache:
                cache.move_to_end(window_index)

    def get_cached_window_for(self, tensor_id: str, window_index: tuple[int, ...]):
        """Get a cached window output, marking as recently used."""
        cache = self._window_cache.get(tensor_id)
        if cache is None:
            return None
        if window_index in cache:
            cache.move_to_end(window_index)
            return cache[window_index]
        return None

    def is_window_cached_for(self, tensor_id: str, window_index: tuple[int, ...]) -> bool:
        """Check if a window is in the cache."""
        cache = self._window_cache.get(tensor_id)
        return cache is not None and window_index in cache

    def get_or_create(self,
                      tensor_id,
                      shape: tuple[int|None, ...],
                      f,
                      output_window,
                      args: tuple = None,
                      args_windows = None,
                      tile_size: int | tuple[int, ...] = 512,
                      dtype=None,
                      batch_size: int | None = None,
                      cache_method: str = 'indirect',
                      cache_limit: int | None = 10 * 1024 * 1024):
        # Local import to avoid circular dependency
        from infinite_tensor.infinite_tensor import InfiniteTensor, DEFAULT_DTYPE
        # Normalize tensor_id to str, generate if missing
        try:
            tid_str = str(tensor_id) if tensor_id is not None else str(uuid.uuid4())
        except Exception:
            tid_str = str(uuid.uuid4())

        # Return existing tensor if already created
        if tid_str in self._tensor_store:
            return self._tensor_store[tid_str]
        
        # Create new tensor
        tensor = InfiniteTensor(
            shape,
            f,
            output_window,
            args=args,
            args_windows=args_windows,
            tile_size=tile_size,
            dtype=(dtype or DEFAULT_DTYPE),
            tile_store=self,
            tensor_id=tid_str,
            _created_via_store=True,
            batch_size=batch_size,
            cache_method=cache_method,
            cache_limit=cache_limit,
        )
        self._tensor_store[tid_str] = tensor
        return tensor

    def get_tensor(self, tensor_id: Any, f = None):
        """Get a tensor by its ID.
        
        Args:
            tensor_id: Tensor identifier
            f: Ignored for MemoryTileStore (kept for interface compatibility)
            
        Returns:
            InfiniteTensor instance
            
        Raises:
            KeyError: If tensor not found
        """
        tid_str = str(tensor_id)
        if tid_str not in self._tensor_store:
            raise KeyError(f"Tensor {tid_str} not found")
        return self._tensor_store[tid_str]


__all__ = ["TileStore", "MemoryTileStore", "HDF5TileStore"]
