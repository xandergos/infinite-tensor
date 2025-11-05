"""Tile storage backends for infinite tensor data.

This module provides the storage abstraction layer for InfinityTensor tiles.
Tiles are chunks of tensor data that fit in memory, and the tile store manages
their lifecycle, retrieval, and cleanup.

The abstract TileStore interface allows for different storage strategies:
- MemoryTileStore: Fast in-memory storage (default)
- HDF5TileStore: Persistent HDF5-backed storage for large datasets
"""

import abc
import uuid
import numpy as np
from typing import Dict, Any, Tuple, Iterable, Set

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

    @abc.abstractmethod
    def get_or_create(self,
                      tensor_id,
                      shape: tuple[int|None, ...],
                      f,
                      output_window,
                      args: tuple = None,
                      args_windows = None,
                      chunk_size: int | tuple[int, ...] = 512,
                      dtype=None):
        """Create or return an InfiniteTensor bound to this store."""
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

    def register_tensor_meta(self, tensor_id: str, meta: dict) -> None:
        self._tensor_meta[tensor_id] = meta
        self._processed_windows_by_tensor[tensor_id] = set()

    def get_tensor_meta(self, tensor_id: str) -> dict:
        return self._tensor_meta[tensor_id]
    
    def clear_tensor(self, tensor_id: str) -> None:
        # Remove tiles
        for tile_idx in list(self.iter_tile_keys_for(tensor_id)):
            self.delete_tile_for(tensor_id, tile_idx)
        # Remove windows
        if tensor_id in self._processed_windows_by_tensor:
            del self._processed_windows_by_tensor[tensor_id]
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

    def get_or_create(self,
                      tensor_id,
                      shape: tuple[int|None, ...],
                      f,
                      output_window,
                      args: tuple = None,
                      args_windows = None,
                      chunk_size: int | tuple[int, ...] = 512,
                      dtype=None):
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
            chunk_size=chunk_size,
            dtype=(dtype or DEFAULT_DTYPE),
            tile_store=self,
            tensor_id=tid_str,
            _created_via_store=True,
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
