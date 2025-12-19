"""HDF5-based persistent tile storage implementation.

This module provides an HDF5-backed tile store for persistent storage of
infinite tensor data. Tiles are stored on disk in HDF5 format, enabling
work with datasets larger than available memory.
"""

import uuid
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Callable, Optional
from collections import OrderedDict
from infinite_tensor.tilestore import TileStore

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    h5py = None


class HDF5TileStore(TileStore):
    """HDF5-based persistent tile storage implementation.
    
    Provides durable storage for large datasets that exceed available memory.
    All data is persisted to disk and can be recovered across sessions.
    
    Args:
        filepath: Path to HDF5 file (created if doesn't exist)
        mode: File mode ('r' for read-only, 'a' for read/write, 'w' to truncate)
        compression: Compression algorithm for tiles ('gzip', 'lzf', None)
        compression_opts: Compression level (0-9 for gzip)
        tile_cache_size: Max tiles in cache (None for unlimited, default 100)
    
    Example:
        >>> store = HDF5TileStore("data/tensors.h5", tile_cache_size=200)
        >>> tensor = store.get_or_create(...)
    """
    
    def __init__(
        self,
        filepath: str | Path,
        mode: str = "a",
        compression: str | None = "gzip",
        compression_opts: int | None = 4,
        tile_cache_size: Optional[int] = 100,
    ):
        """Initialize HDF5 tile store.
        
        Args:
            filepath: Path to HDF5 file
            mode: File open mode ('r', 'a', 'w')
            compression: Compression algorithm for tile datasets
            compression_opts: Compression options (level for gzip)
            tile_cache_size: Maximum number of tiles to cache in memory.
                           Set to None for unlimited cache. Default is 100.
            
        Raises:
            ImportError: If h5py is not installed
        """
        if not HAS_H5PY:
            raise ImportError(
                "h5py is required for HDF5TileStore. "
                "Install it with: pip install infinite-tensor[hdf5]"
            )
        super().__init__()
        self.filepath = Path(filepath)
        self.compression = compression
        self.compression_opts = compression_opts
        self.tile_cache_size = tile_cache_size
        
        # Handle mode: 'w' truncates once, then becomes 'a'
        if mode == 'w':
            # Truncate file by opening in 'w' mode
            with h5py.File(self.filepath, 'w') as f:
                f.create_group("tensors")
            self.mode = 'a'  # Switch to append for subsequent operations
        elif mode == 'r':
            self.mode = 'r'  # Read-only
        else:
            self.mode = 'a'  # Append mode (default)
        
        # Single file handle reused across operations (single-threaded usage)
        self._file: Optional["h5py.File"] = None
        
        # LRU cache for tiles only (the big data) - key is (tensor_id, tile_index)
        self._tile_cache: OrderedDict[tuple, Any] = OrderedDict()
        
        # Unlimited caches for everything else (small metadata)
        self._function_cache: Dict[str, Callable] = {}
        self._tensor_cache: Dict[str, Any] = {}  # Tensor instances
        self._metadata_cache: Dict[str, dict] = {}  # Tensor metadata
        self._processed_windows_cache: Dict[str, set] = {}  # Processed windows per tensor
        
        # Window cache for direct caching (per-tensor LRU cache, stored in memory)
        self._window_cache: Dict[str, OrderedDict] = {}
        self._window_cache_size: Dict[str, int] = {}
        
        # Initialize file structure (only if not read-only)
        if self.mode != 'r':
            self._ensure_file_exists()

        self._open_file()
    
    def _ensure_file_exists(self) -> None:
        """Create file and root structure if it doesn't exist."""
        with h5py.File(self.filepath, "a") as f:
            if "tensors" not in f:
                f.create_group("tensors")
    
    def _cache_tile(self, tensor_id: str, tile_index: tuple[int, ...], tile) -> None:
        """Add tile to LRU cache, evicting oldest if cache is full.
        
        Args:
            tensor_id: Tensor identifier
            tile_index: Tile coordinate tuple
            tile: InfinityTensorTile instance to cache
        """
        key = (tensor_id, tile_index)
        
        # Move to end if already exists (mark as recently used)
        if key in self._tile_cache:
            self._tile_cache.move_to_end(key)
            return
        
        # Add new tile
        self._tile_cache[key] = tile
        
        # Evict oldest if cache is full
        if self.tile_cache_size is not None and len(self._tile_cache) > self.tile_cache_size:
            self._tile_cache.popitem(last=False)
    
    def _get_cached_tile(self, tensor_id: str, tile_index: tuple[int, ...]):
        """Get tile from cache, marking as recently used.
        
        Args:
            tensor_id: Tensor identifier
            tile_index: Tile coordinate tuple
            
        Returns:
            InfinityTensorTile instance or None if not cached
        """
        key = (tensor_id, tile_index)
        if key in self._tile_cache:
            # Move to end (mark as recently used)
            self._tile_cache.move_to_end(key)
            return self._tile_cache[key]
        return None
    
    def _remove_cached_tile(self, tensor_id: str, tile_index: tuple[int, ...]) -> None:
        """Remove a tile from cache if present.
        
        Args:
            tensor_id: Tensor identifier
            tile_index: Tile coordinate tuple
        """
        key = (tensor_id, tile_index)
        if key in self._tile_cache:
            del self._tile_cache[key]
    
    def _open_file(self) -> None:
        """Open the underlying HDF5 file handle if it is not already open."""
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
    
    def close(self) -> None:
        """Close all file handles. Call when done with the store."""
        self._close_file()
    
    def flush(self) -> None:
        """Flush pending data to disk if the file is open."""
        if self._file is not None:
            self._file.flush()

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass
    
    def __enter__(self):
        """Context manager support."""
        self._open_file()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.close()
    
    @staticmethod
    def _encode_tile_index(tile_index: tuple[int, ...]) -> str:
        """Encode tuple tile index as string for HDF5 group name.
        
        Args:
            tile_index: Tuple of integers
            
        Returns:
            String representation: "0_1_2" for (0, 1, 2)
        """
        return "_".join(map(str, tile_index))
    
    @staticmethod
    def _decode_tile_index(encoded: str) -> tuple[int, ...]:
        """Decode string back to tuple tile index.
        
        Args:
            encoded: String like "0_1_2"
            
        Returns:
            Tuple like (0, 1, 2)
        """
        return tuple(map(int, encoded.split("_")))
    
    def _get_tensor_group(self, f, tensor_id: str, create: bool = False):
        """Get or create tensor group in HDF5 file.
        
        Args:
            f (h5py.File): HDF5 file handle
            tensor_id: Tensor identifier
            create: Whether to create group if it doesn't exist
            
        Returns:
            h5py.Group | None: HDF5 group or None if not found and create=False
        """
        path = f"tensors/{tensor_id}"
        if path in f:
            return f[path]
        elif create:
            group = f.create_group(path)
            # Initialize subgroups
            group.create_group("tiles")
            group.create_group("metadata")
            return group
        return None
    
    def register_tensor_meta(self, tensor_id: str, meta: dict) -> None:
        """Register a new tensor and its metadata in the store.
        
        Args:
            tensor_id: Unique tensor identifier
            meta: Metadata dictionary containing tensor configuration
        """
        # Cache metadata
        self._metadata_cache[tensor_id] = meta
        self._processed_windows_cache.pop(tensor_id, None)
        
        # Persist to disk
        f = self._get_file()
        tensor_group = self._get_tensor_group(f, tensor_id, create=True)
        metadata_group = tensor_group["metadata"]
        
        # Store metadata as JSON in attributes
        json_dump = json.dumps(meta)
        if metadata_group.attrs.get("meta") != json_dump:
            metadata_group.attrs["meta"] = json_dump
        
        # Create empty processed windows dataset
        if "processed_windows" not in tensor_group:
            # Use variable-length string dtype for window indices
            dt = h5py.special_dtype(vlen=str)
            tensor_group.create_dataset(
                "processed_windows",
                shape=(0,),
                maxshape=(None,),
                dtype=dt,
            )
        
        f.flush()
    
    def get_tensor_meta(self, tensor_id: str) -> dict:
        """Fetch metadata dict for a tensor.
        
        Args:
            tensor_id: Tensor identifier
            
        Returns:
            Metadata dictionary
            
        Raises:
            KeyError: If tensor not found
        """
        # Check cache first
        if tensor_id in self._metadata_cache:
            return self._metadata_cache[tensor_id]
        
        # Load from disk and cache
        f = self._get_file()
        tensor_group = self._get_tensor_group(f, tensor_id)
        if tensor_group is None:
            raise KeyError(f"Tensor {tensor_id} not found")
        
        metadata_group = tensor_group["metadata"]
        meta_json = metadata_group.attrs.get("meta")
        if meta_json is None:
            raise KeyError(f"No metadata for tensor {tensor_id}")
        
        meta = json.loads(meta_json)
        self._metadata_cache[tensor_id] = meta
        return meta
    
    def clear_tensor(self, tensor_id: str) -> None:
        """Remove all state for a tensor (tiles, windows, metadata).
        
        Args:
            tensor_id: Tensor identifier
        """
        # Remove from all caches
        keys_to_remove = [key for key in self._tile_cache.keys() if key[0] == tensor_id]
        for key in keys_to_remove:
            del self._tile_cache[key]
        
        if tensor_id in self._function_cache:
            del self._function_cache[tensor_id]
        if tensor_id in self._tensor_cache:
            del self._tensor_cache[tensor_id]
        if tensor_id in self._metadata_cache:
            del self._metadata_cache[tensor_id]
        if tensor_id in self._processed_windows_cache:
            del self._processed_windows_cache[tensor_id]
        
        # Remove from disk
        f = self._get_file()
        
        # Remove tensor group
        tensor_path = f"tensors/{tensor_id}"
        if tensor_path in f:
            del f[tensor_path]
        
        f.flush()
    
    def _reconstruct_tensor_from_metadata(self, tensor_id: str, f: Callable, batch_size: int | None = None,
                                          cache_method: str = 'indirect', cache_limit: int | None = 10 * 1024 * 1024) -> Any:
        """Reconstruct an InfiniteTensor from stored metadata and a provided function.
        
        Args:
            tensor_id: Tensor identifier
            f: The computation function for this tensor
            cache_method: Caching strategy (default from caller or stored metadata)
            cache_limit: Cache limit for direct caching
            
        Returns:
            InfiniteTensor instance
            
        Raises:
            KeyError: If tensor metadata not found
        """
        # Import locally to avoid circular dependency
        from infinite_tensor.infinite_tensor import InfiniteTensor, _str_to_dtype
        from infinite_tensor.tensor_window import TensorWindow
        
        # Load metadata
        meta = self.get_tensor_meta(tensor_id)
        
        # Deserialize metadata
        shape = tuple(None if x is None else int(x) for x in meta['shape'])
        tile_size = tuple(int(x) for x in meta.get('tile_size', meta.get('chunk_size')))
        dtype = _str_to_dtype(meta['dtype'])
        
        # Use stored cache_method/cache_limit if available, otherwise use defaults
        stored_cache_method = meta.get('cache_method', cache_method)
        stored_cache_limit = meta.get('cache_limit', cache_limit)
        
        # Reconstruct args (will be tensor IDs, need to get actual tensors)
        arg_ids = meta['args']
        args = tuple(self.get_tensor(arg_id) for arg_id in arg_ids) if arg_ids else None
        
        # Reconstruct windows
        output_window = TensorWindow.from_dict(meta['output_window'])
        args_windows = [TensorWindow.from_dict(w) if w is not None else None 
                       for w in meta['args_windows']] if meta.get('args_windows') else None
        
        # Create tensor instance
        tensor = InfiniteTensor(
            shape=shape,
            f=f,
            output_window=output_window,
            args=args,
            args_windows=args_windows,
            tile_size=tile_size,
            dtype=dtype,
            tile_store=self,
            tensor_id=tensor_id,
            _created_via_store=True,
            batch_size=batch_size,
            cache_method=stored_cache_method,
            cache_limit=stored_cache_limit,
        )
        
        return tensor
    
    def get_tile_for(self, tensor_id: str, tile_index: tuple[int, ...]):
        """Get tile data for a specific tensor and tile index.
        
        Args:
            tensor_id: Tensor identifier
            tile_index: Tile coordinate tuple
            
        Returns:
            InfinityTensorTile or None if not found
        """
        # Check cache first
        cached_tile = self._get_cached_tile(tensor_id, tile_index)
        if cached_tile is not None:
            return cached_tile
        
        # Load from disk
        f = self._get_file()
        tensor_group = self._get_tensor_group(f, tensor_id)
        if tensor_group is None:
            return None
        
        tiles_group = tensor_group["tiles"]
        encoded_idx = self._encode_tile_index(tile_index)
        
        if encoded_idx not in tiles_group:
            return None
        
        # Load tile data
        tile_dataset = tiles_group[encoded_idx]
        values = tile_dataset[:]
        
        # Import locally to avoid circular dependency
        from infinite_tensor.infinite_tensor import InfinityTensorTile
        import torch
        
        # Convert numpy array back to torch tensor
        tile = InfinityTensorTile(values=torch.from_numpy(values))
        
        # Cache the loaded tile
        self._cache_tile(tensor_id, tile_index, tile)
        
        return tile
    
    def set_tile_for(self, tensor_id: str, tile_index: tuple[int, ...], value) -> None:
        """Set tile data for a specific tensor and tile index.
        
        Args:
            tensor_id: Tensor identifier
            tile_index: Tile coordinate tuple
            value: InfinityTensorTile object to store
        """
        # Update cache
        self._cache_tile(tensor_id, tile_index, value)
        
        # Write to disk
        f = self._get_file()
        tensor_group = self._get_tensor_group(f, tensor_id, create=True)
        tiles_group = tensor_group["tiles"]
        encoded_idx = self._encode_tile_index(tile_index)
        
        # Extract data from InfinityTensorTile
        from infinite_tensor.infinite_tensor import InfinityTensorTile
        if isinstance(value, InfinityTensorTile):
            tile_data = value.values
        else:
            # Fallback for raw arrays (shouldn't happen in normal usage)
            tile_data = value
        
        # Convert torch tensor to numpy for HDF5 storage
        import torch
        if isinstance(tile_data, torch.Tensor):
            tile_data = tile_data.cpu().numpy()
        
        # Delete existing dataset if present
        if encoded_idx in tiles_group:
            del tiles_group[encoded_idx]
        
        # Create new dataset with compression
        dataset = tiles_group.create_dataset(
            encoded_idx,
            data=tile_data,
            compression=self.compression,
            compression_opts=self.compression_opts,
        )
        f.flush()
    
    def delete_tile_for(self, tensor_id: str, tile_index: tuple[int, ...]) -> None:
        """Delete tile data for a specific tensor and tile index.
        
        Args:
            tensor_id: Tensor identifier
            tile_index: Tile coordinate tuple
        """
        # Remove from cache
        print(f"Deleting tile for {tensor_id} at {tile_index}")
        self._remove_cached_tile(tensor_id, tile_index)
        
        # Remove from disk
        f = self._get_file()
        tensor_group = self._get_tensor_group(f, tensor_id)
        if tensor_group is None:
            return
        
        tiles_group = tensor_group["tiles"]
        encoded_idx = self._encode_tile_index(tile_index)
        
        if encoded_idx in tiles_group:
            del tiles_group[encoded_idx]
            f.flush()
    
    def iter_tile_keys_for(self, tensor_id: str) -> Iterable[tuple[int, ...]]:
        """Iterate over tile indices for a tensor.
        
        Args:
            tensor_id: Tensor identifier
            
        Yields:
            Tile index tuples
        """
        f = self._get_file()
        tensor_group = self._get_tensor_group(f, tensor_id)
        if tensor_group is None:
            return
        
        tiles_group = tensor_group["tiles"]
        for encoded_idx in tiles_group.keys():
            yield self._decode_tile_index(encoded_idx)
    
    def is_window_processed_for(self, tensor_id: str, window_index: tuple[int, ...]) -> bool:
        """Check if a window has been processed for a tensor.
        
        Args:
            tensor_id: Tensor identifier
            window_index: Window coordinate tuple
            
        Returns:
            True if window has been processed
        """
        # Check cache first
        if tensor_id in self._processed_windows_cache:
            return window_index in self._processed_windows_cache[tensor_id]
        
        # Load from disk and cache
        f = self._get_file()
        tensor_group = self._get_tensor_group(f, tensor_id)
        if tensor_group is None:
            self._processed_windows_cache[tensor_id] = set()
            return False
        
        if "processed_windows" not in tensor_group:
            self._processed_windows_cache[tensor_id] = set()
            return False
        
        pw_dataset = tensor_group["processed_windows"]
        
        # Load all processed windows into cache
        stored_windows = [w.decode() if isinstance(w, bytes) else w for w in pw_dataset[:]]
        windows_set = {self._decode_tile_index(w) for w in stored_windows}
        self._processed_windows_cache[tensor_id] = windows_set
        
        return window_index in windows_set
    
    def mark_window_processed_for(self, tensor_id: str, window_index: tuple[int, ...]) -> None:
        """Mark a window as processed for a tensor.
        
        Args:
            tensor_id: Tensor identifier
            window_index: Window coordinate tuple
        """
        # If not in cache, load from disk first to avoid partial cache
        if tensor_id not in self._processed_windows_cache:
            f = self._get_file()
            tensor_group = self._get_tensor_group(f, tensor_id, create=True)
            
            if "processed_windows" in tensor_group:
                pw_dataset = tensor_group["processed_windows"]
                stored_windows = [w.decode() if isinstance(w, bytes) else w for w in pw_dataset[:]]
                windows_set = {self._decode_tile_index(w) for w in stored_windows}
                self._processed_windows_cache[tensor_id] = windows_set
            else:
                self._processed_windows_cache[tensor_id] = set()
        
        # Check if already in cache
        assert window_index not in self._processed_windows_cache[tensor_id]
        
        # Update cache
        self._processed_windows_cache[tensor_id].add(window_index)
        
        # Persist to disk
        f = self._get_file()
        tensor_group = self._get_tensor_group(f, tensor_id, create=True)
        
        if "processed_windows" not in tensor_group:
            dt = h5py.special_dtype(vlen=str)
            tensor_group.create_dataset(
                "processed_windows",
                shape=(0,),
                maxshape=(None,),
                dtype=dt,
            )
        
        pw_dataset = tensor_group["processed_windows"]
        encoded = self._encode_tile_index(window_index)
        
        # Check if already present on disk (decode bytes for comparison)
        stored_windows = [w.decode() if isinstance(w, bytes) else w for w in pw_dataset[:]]
        if encoded in stored_windows:
            return
        
        # Append new window
        current_size = pw_dataset.shape[0]
        pw_dataset.resize((current_size + 1,))
        pw_dataset[current_size] = encoded
        
        f.flush()

    def _tensor_size_bytes(self, t) -> int:
        """Calculate the size of a tensor in bytes."""
        return t.numel() * t.element_size()

    def cache_window_for(self, tensor_id: str, window_index: tuple[int, ...], output, cache_limit: Optional[int]) -> None:
        """Cache a window output with LRU eviction based on bytes (stored in memory)."""
        if tensor_id not in self._window_cache:
            self._window_cache[tensor_id] = OrderedDict()
            self._window_cache_size[tensor_id] = 0
        
        cache = self._window_cache[tensor_id]
        
        if window_index in cache:
            cache.move_to_end(window_index)
            return
        
        output_bytes = self._tensor_size_bytes(output)
        cache[window_index] = output
        self._window_cache_size[tensor_id] += output_bytes
        
        # Evict oldest entries until under limit (if limit is set)
        if cache_limit is not None:
            while self._window_cache_size[tensor_id] > cache_limit and len(cache) > 1:
                _, evicted = cache.popitem(last=False)
                self._window_cache_size[tensor_id] -= self._tensor_size_bytes(evicted)

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
    
    def get_or_create(
        self,
        tensor_id,
        shape: tuple[int | None, ...],
        f,
        output_window,
        args: tuple = None,
        args_windows=None,
        tile_size: int | tuple[int, ...] = 512,
        dtype=None,
        batch_size: int | None = None,
        cache_method: str = 'indirect',
        cache_limit: int | None = 10 * 1024 * 1024,
    ):
        """Create or return an InfiniteTensor bound to this store.
        
        Args:
            tensor_id: Unique identifier for the tensor
            shape: Tensor shape (None for infinite dimensions)
            f: Computation function
            output_window: Output window specification
            args: Argument tensors
            args_windows: Argument window specifications
            tile_size: Tile size
            dtype: Data type
            cache_method: Caching strategy - 'indirect' (default) stores tiles, 
                         'direct' caches window outputs directly.
            cache_limit: Maximum cache size in bytes for direct caching (default: 10MB).
                        None for unlimited cache.
            
        Returns:
            InfiniteTensor instance
        """
        # Local import to avoid circular dependency
        from infinite_tensor.infinite_tensor import InfiniteTensor, DEFAULT_DTYPE
        
        # Normalize tensor_id to str, generate if missing
        tid_str = str(tensor_id) if tensor_id is not None else str(uuid.uuid4())
        
        # Check tensor cache first
        if tid_str in self._tensor_cache:
            return self._tensor_cache[tid_str]
        
        # Check if tensor exists in file
        f_handle = self._get_file()
        tensor_exists = self._get_tensor_group(f_handle, tid_str) is not None
        
        if tensor_exists:
            # Reconstruct from metadata with provided function
            tensor = self._reconstruct_tensor_from_metadata(tid_str, f, batch_size=batch_size,
                                                            cache_method=cache_method, cache_limit=cache_limit)
            # Cache tensor and function
            self._tensor_cache[tid_str] = tensor
            self._function_cache[tid_str] = f
            return tensor
        
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
        
        # Cache tensor and function
        self._tensor_cache[tid_str] = tensor
        self._function_cache[tid_str] = f
        
        return tensor
    
    def get_tensor(self, tensor_id: Any, f: Optional[Callable] = None):
        """Get a tensor by its ID.
        
        Args:
            tensor_id: Tensor identifier
            f: Optional computation function. Required if tensor was not created
               this session and needs to be reconstructed from metadata.
            
        Returns:
            InfiniteTensor instance
            
        Raises:
            KeyError: If tensor not found in metadata
            ValueError: If tensor needs reconstruction but no function provided
        """
        tid_str = str(tensor_id)
        
        # Check tensor cache first
        if tid_str in self._tensor_cache:
            return self._tensor_cache[tid_str]
        
        # Check if tensor exists in file
        f_handle = self._get_file()
        if self._get_tensor_group(f_handle, tid_str) is None:
            raise KeyError(f"Tensor {tid_str} not found")
        
        # Reconstruct from metadata
        # First try function cache (for tensors created this session)
        func = f or self._function_cache.get(tid_str)
        if func is None:
            raise ValueError(
                f"Tensor {tid_str} not found in function cache and no function provided. "
                "Please provide the computation function to reconstruct the tensor."
            )
        
        tensor = self._reconstruct_tensor_from_metadata(tid_str, func)
        
        # Cache tensor and function
        self._tensor_cache[tid_str] = tensor
        if f is not None:
            self._function_cache[tid_str] = f
        
        return tensor
