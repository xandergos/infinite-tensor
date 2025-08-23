"""Tile storage backends for infinite tensor data.

This module provides the storage abstraction layer for InfinityTensor tiles.
Tiles are chunks of tensor data that fit in memory, and the tile store manages
their lifecycle, retrieval, and cleanup.

The abstract TileStore interface allows for different storage strategies:
- MemoryTileStore: Fast in-memory storage (default)
- Future: DiskTileStore, DistributedTileStore, etc.
"""

import abc
import os
import h5py
import numpy as np
from typing import List, Optional
from uuid import UUID

class TileStore(abc.ABC):
    """Abstract base class for tile storage backends.
    
    TileStore defines the interface for storing, retrieving, and managing
    InfinityTensorTile objects. Different implementations can provide
    various storage strategies (memory, disk, distributed, etc.).
    
    The tile store also tracks window processing state to prevent
    duplicate computation and enable efficient memory management.
    """
    @abc.abstractmethod
    def get(self, key: tuple[int, ...]):
        """Retrieve a tile from storage.

        Args:
            key: Unique identifier for the tile, typically (tensor_uuid, tile_index)
            
        Returns:
            InfinityTensorTile object if found, None if not present
            
        Notes:
            Implementation should be thread-safe if used in concurrent contexts.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def set(self, key: tuple[int, ...], value):
        """Store a tile in the storage backend.

        Args:
            key: Unique identifier for the tile
            value: InfinityTensorTile object to store
            
        Notes:
            Should overwrite existing tiles with the same key.
            Implementation should handle memory management appropriately.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def delete(self, key: tuple[int, ...]):
        """Remove a tile from storage.

        Args:
            key: Unique identifier of the tile to delete
            
        Notes:
            Should handle deletion of non-existent keys gracefully.
            This is called during cleanup to free memory.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def keys(self) -> List[tuple[int, ...]]:
        """Get all tile keys currently in storage.
        
        Returns:
            List of all keys (tile identifiers) currently stored
            
        Notes:
            Used for cleanup operations and memory management.
            Should return a snapshot of keys at call time.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def is_window_processed(self, window_index: tuple[int, ...]) -> bool:
        """Check if a window has already been processed.
        
        Args:
            window_index: N-dimensional index of the window to check
            
        Returns:
            True if the window has been processed, False otherwise
            
        Notes:
            Used to prevent duplicate computation of the same window.
            Critical for performance in overlapping window scenarios.
        """
    
    @abc.abstractmethod
    def mark_window_processed(self, window_index: tuple[int, ...]):
        """Mark a window as having been processed.
        
        Args:
            window_index: N-dimensional index of the processed window
            
        Notes:
            Called after successful window processing to prevent recomputation.
            Should be persistent for the lifetime of the tensor.
        """

class MemoryTileStore(TileStore):
    """In-memory tile storage implementation.
    
    This is the default tile store that keeps all tiles in RAM using
    Python dictionaries. Provides fast access but is limited by available
    memory. Suitable for moderate-sized datasets or when disk I/O should
    be avoided.
    
    Attributes:
        store: Dictionary mapping tile keys to InfinityTensorTile objects
        processed_windows: Set of window indices that have been processed
        dependent_windows: Dictionary for tracking window dependencies (unused in current implementation)
    """
    def __init__(self):
        """Initialize an empty in-memory tile store."""
        super().__init__()
        self.store = {}  # tile_key -> InfinityTensorTile
        self.processed_windows = set()  # window_index tuples that have been processed
        self.dependent_windows = {}  # Reserved for future dependency tracking features
    
    def get(self, key: tuple[int, ...]):
        """Retrieve a tile from memory.
        
        Args:
            key: Tile identifier
            
        Returns:
            InfinityTensorTile if found, None otherwise
        """
        return self.store.get(key)

    def set(self, key: tuple[int, ...], value):
        """Store a tile in memory.
        
        Args:
            key: Tile identifier  
            value: InfinityTensorTile to store
        """
        self.store[key] = value
        
    def delete(self, key: tuple[int, ...]):
        """Remove a tile from memory.
        
        Args:
            key: Tile identifier to delete
            
        Notes:
            Silently handles deletion of non-existent keys.
        """
        if key in self.store:
            del self.store[key]
    
    def keys(self) -> List[tuple[int, ...]]:
        """Get all tile keys currently in memory.
        
        Returns:
            List of all tile identifiers currently stored
        """
        return list(self.store.keys())
    
    def is_window_processed(self, window_index: tuple[int, ...]) -> bool:
        """Check if a window has been processed.
        
        Args:
            window_index: Window coordinates to check
            
        Returns:
            True if window was already processed
        """
        return window_index in self.processed_windows
    
    def mark_window_processed(self, window_index: tuple[int, ...]):
        """Mark a window as processed.
        
        Args:
            window_index: Window coordinates that have been processed
        """
        self.processed_windows.add(window_index)


class HDF5TileStore(TileStore):
    """HDF5-based tile storage implementation.
    
    This tile store persists tiles to disk using HDF5 format, providing a balance
    between performance and memory usage. Suitable for large datasets that don't
    fit in memory.
    
    The HDF5 file structure is organized as follows:
    - /tiles/: Group containing all tile datasets
        - Each tile is stored as a dataset named by its key
    - /windows/: Group containing window processing state
        - Stored as a dataset of processed window indices
    
    Attributes:
        file_path: Path to the HDF5 file
        mode: File access mode ('w' for overwrite, 'a' for append)
    """
    def __init__(self, file_path: str, mode: str = 'a'):
        """Initialize HDF5 tile store.
        
        Args:
            file_path: Path where the HDF5 file should be stored
            mode: File access mode ('w' for overwrite, 'a' for append)
        """
        super().__init__()
        self.file_path = file_path
        self.mode = mode
        
        # Create file and required groups if they don't exist
        if mode == 'a' and not os.path.exists(file_path):
            with h5py.File(file_path, 'w') as f:
                f.create_group('tiles')
                f.create_group('windows')
                # Create dataset for processed windows
                f.create_dataset('windows/processed', 
                               shape=(0, 0),  # Will be resized as needed
                               maxshape=(None, None),
                               dtype=np.int64)
    
    def _key_to_path(self, key: tuple[int, ...]) -> str:
        """Convert a tile key to an HDF5 dataset path.
        
        Args:
            key: Tile identifier tuple
            
        Returns:
            String path for the tile's dataset in the HDF5 file
        """
        return f"tiles/{'_'.join(str(k) for k in key)}"
    
    def get(self, key: tuple[int, ...]):
        """Retrieve a tile from HDF5 storage.
        
        Args:
            key: Tile identifier
            
        Returns:
            InfinityTensorTile if found, None otherwise
        """
        try:
            with h5py.File(self.file_path, self.mode) as f:
                path = self._key_to_path(key)
                if path in f:
                    return np.array(f[path])
                return None
        except (OSError, KeyError):
            return None
    
    def set(self, key: tuple[int, ...], value):
        """Store a tile in HDF5.
        
        Args:
            key: Tile identifier
            value: InfinityTensorTile to store
        """
        if self.mode == 'r':
            raise ValueError("Cannot write to read-only HDF5 store")
            
        with h5py.File(self.file_path, 'a') as f:
            path = self._key_to_path(key)
            if path in f:
                del f[path]  # Delete existing dataset
            f.create_dataset(path, data=value)
    
    def delete(self, key: tuple[int, ...]):
        """Remove a tile from HDF5 storage.
        
        Args:
            key: Tile identifier to delete
        """
        if self.mode == 'r':
            return
            
        try:
            with h5py.File(self.file_path, 'a') as f:
                path = self._key_to_path(key)
                if path in f:
                    del f[path]
        except OSError:
            pass  # Ignore errors when deleting
    
    def keys(self) -> List[tuple[int, ...]]:
        """Get all tile keys currently in storage.
        
        Returns:
            List of all tile identifiers currently stored
        """
        keys = []
        try:
            with h5py.File(self.file_path, self.mode) as f:
                if 'tiles' in f:
                    for name in f['tiles'].keys():
                        # Convert dataset name back to key tuple
                        key = tuple(int(k) for k in name.split('_'))
                        keys.append(key)
        except OSError:
            pass
        return keys
    
    def is_window_processed(self, window_index: tuple[int, ...]) -> bool:
        """Check if a window has been processed.
        
        Args:
            window_index: Window coordinates to check
            
        Returns:
            True if window was already processed
        """
        try:
            with h5py.File(self.file_path, self.mode) as f:
                if 'windows/processed' not in f:
                    return False
                    
                processed = f['windows/processed']
                if processed.shape[1] == 0:  # No windows processed yet
                    return False
                    
                # Convert window index to numpy array for comparison
                window_arr = np.array(window_index)
                
                # Check if window exists in processed dataset
                for i in range(processed.shape[0]):
                    if np.array_equal(processed[i, :], window_arr):
                        return True
                return False
        except OSError:
            return False
    
    def mark_window_processed(self, window_index: tuple[int, ...]):
        """Mark a window as processed.
        
        Args:
            window_index: Window coordinates that have been processed
        """
        if self.mode == 'r':
            return
            
        try:
            with h5py.File(self.file_path, 'a') as f:
                processed = f['windows/processed']
                
                # Initialize or resize dataset if needed
                if processed.shape[1] == 0:
                    processed.resize((1, len(window_index)))
                    processed[0] = window_index
                else:
                    # Check if window is already marked
                    if not self.is_window_processed(window_index):
                        processed.resize((processed.shape[0] + 1, processed.shape[1]))
                        processed[-1] = window_index
        except OSError:
            pass  # Ignore errors when marking windows
