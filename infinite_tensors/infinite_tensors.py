from dataclasses import dataclass, field
from functools import lru_cache
import gc
import logging
import math
import os
import random
import tempfile
import time
from typing import Any, Callable, Optional, Union
import weakref
import numpy as np
import torch
import h5py
from collections import OrderedDict
import itertools
import uuid
from uuid import UUID

from infinite_tensors.tensor_window import TensorWindow
from infinite_tensors.tilestore import MemoryTileStore, TileStore
from infinite_tensors.utils import normalize_slice, standardize_indices

# COORDINATE SYSTEM DEFINITIONS:
# pixel space - Pixel space with all dimensions included (raw tensor coordinates)
# window space - Each point is a window of pixels (used for sliding window operations)
# tile space - Each point is a tile of pixels (internal chunking for memory management)

# CONSTANTS
DEFAULT_CHUNK_SIZE = 512
DEFAULT_DTYPE = torch.float32

# ERROR MESSAGES
DEPENDENCY_ERROR_MSG = "Setting values on infinite tensors with dependencies is not supported, as the tensor's state may be inconsistent. Please set values before creating any dependent tensors."
TILE_DELETED_ERROR_MSG = "Tile has been deleted. This indicates either a bug or an attempt to access a tensor after cleanup."
SHAPE_MISMATCH_ERROR_MSG = "Value shape {actual} does not match indexed shape {expected}"
OUTPUT_SHAPE_ERROR_MSG = "Function output shape {actual} does not match expected window shape {expected}"
DEVICE_MISMATCH_ERROR_MSG = "Device mismatch: value is on {actual}, but infinite tensors require CPU tensors."

@dataclass
class TensorConfig:
    """Configuration for InfiniteTensor behavior.
    
    This centralizes configuration options and makes it easier to add
    new configuration parameters without changing the constructor signature.
    """
    chunk_size: Union[int, tuple[int, ...]] = DEFAULT_CHUNK_SIZE
    dtype: torch.dtype = DEFAULT_DTYPE
    enable_logging: bool = False
    memory_cleanup_threshold: float = 0.8  # Cleanup when memory usage exceeds this fraction
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if isinstance(self.chunk_size, int) and self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if isinstance(self.chunk_size, tuple) and any(c <= 0 for c in self.chunk_size):
            raise ValueError("All chunk sizes must be positive")

# Set up logging
logger = logging.getLogger(__name__)

# CUSTOM EXCEPTIONS
class InfiniteTensorError(Exception):
    """Base exception for infinite tensor operations."""
    pass

class TileAccessError(InfiniteTensorError):
    """Raised when trying to access a deleted or invalid tile."""
    pass

class DependencyError(InfiniteTensorError):
    """Raised when operations conflict with tensor dependencies."""
    pass

class ShapeMismatchError(InfiniteTensorError):
    """Raised when tensor shapes don't match expected dimensions."""
    pass

class ValidationError(InfiniteTensorError):
    """Raised when parameter validation fails."""
    pass

def _validate_shape(shape: tuple) -> None:
    """Validate tensor shape specification."""
    if not isinstance(shape, tuple):
        raise ValidationError(f"Shape must be a tuple, got {type(shape)}")
    if len(shape) == 0:
        raise ValidationError("Shape cannot be empty")
    for i, dim in enumerate(shape):
        if dim is not None and (not isinstance(dim, int) or dim <= 0):
            raise ValidationError(f"Dimension {i} must be None or positive integer, got {dim}")

def _validate_chunk_size(chunk_size: Union[int, tuple], infinite_dims: int) -> None:
    """Validate chunk size parameter."""
    if isinstance(chunk_size, int):
        if chunk_size <= 0:
            raise ValidationError(f"Chunk size must be positive, got {chunk_size}")
    elif isinstance(chunk_size, tuple):
        if len(chunk_size) != infinite_dims:
            raise ValidationError(f"Chunk size tuple length {len(chunk_size)} must match infinite dimensions {infinite_dims}")
        if any(c <= 0 for c in chunk_size):
            raise ValidationError(f"All chunk sizes must be positive, got {chunk_size}")
    else:
        raise ValidationError(f"Chunk size must be int or tuple, got {type(chunk_size)}")

def _validate_function(f: Callable) -> None:
    """Validate the generating function."""
    if not callable(f):
        raise ValidationError(f"Function must be callable, got {type(f)}")

def _validate_window_args(args: tuple, args_windows, kwargs: dict, kwargs_windows) -> None:
    """Validate window argument consistency."""
    if args_windows is not None and len(args_windows) != len(args):
        raise ValidationError(f"args_windows length {len(args_windows)} must match args length {len(args)}")
    
    if kwargs_windows is not None:
        for key in kwargs_windows:
            if key not in kwargs:
                raise ValidationError(f"kwargs_windows key '{key}' not found in kwargs")

@dataclass
class InfinityTensorTile:
    """A single tile storing a chunk of tensor data.
    
    This represents a contiguous chunk of an infinite tensor that fits in memory.
    Each tile tracks how many dependency windows have been processed to enable
    automatic cleanup when the tile is no longer needed.
    
    Attributes:
        values: The actual tensor data for this tile
        dependency_windows_processed: Counter tracking how many dependency windows 
                                    that reference this tile have been processed.
                                    Used for automatic memory management.
    """
    values: torch.Tensor
    dependency_windows_processed: int = 0

class InfiniteTensor:
    def __init__(self, 
                 shape: tuple[int|None, ...],
                 f: Callable,
                 output_window: TensorWindow,
                 args: tuple = None,
                 kwargs: dict = None,
                 args_windows = None,
                 kwargs_windows = None,
                 chunk_size: Union[int, tuple[int, ...]] = DEFAULT_CHUNK_SIZE,
                 dtype: torch.dtype = DEFAULT_DTYPE,
                 tile_store: TileStore = None):
        """Initialize an InfiniteTensor.

        An InfiniteTensor represents a theoretically infinite tensor that is processed in chunks.
        Operations can be performed on the tensor in a sliding window manner without loading
        the entire tensor into memory.

        Args:
            shape: Shape of the tensor. Use None to indicate that the tensor is infinite in that dimension. For example (3, 10, None, None) 
                   indicates a 4-dimensional tensor with the first two dimensions of size 3 and 10, and the last two dimensions being infinite.
                   This might be used to represent a batch of 3 images, with 10 channels and an infinite width and height.
            f: Callable[[Any, *Any, **Any], torch.Tensor]: A function that takes a context and optional arguments and returns a tensor. 
               The function signature should be f(ctx, *args, **kwargs), where ctx is the index of the window that is being processed.
            args: Positional arguments to pass to the function f.
            kwargs: Keyword arguments to pass to the function f.
            args_windows: Optional positional arguments specific to window processing.
            kwargs_windows: Optional keyword arguments specific to window processing.
            chunk_size: Size of each chunk. Can be an integer for uniform chunk size, or tuple of integers to specify a different chunk size for each dimension.
                        The number of 'None' values in 'shape' must match the number of dimensions in 'chunk_size' if it is a tuple.
            dtype: PyTorch data type for the tensor (default: torch.float32)
            tile_store: TileStore to use for storing tiles. If None, tiles are stored in memory.
        """
        # Validate parameters
        _validate_shape(shape)
        _validate_function(f)
        
        infinite_dims = sum(1 for dim in shape if dim is None)
        _validate_chunk_size(chunk_size, infinite_dims)
        _validate_window_args(args or (), args_windows, kwargs or {}, kwargs_windows or {})
        
        self._shape = shape
        if isinstance(chunk_size, int):
            self._chunk_size = (chunk_size,) * infinite_dims
        else:
            self._chunk_size = chunk_size
        self._dtype = dtype
        self._uuid = uuid.uuid4()
        self._operators = []
        self._store = tile_store or MemoryTileStore()
        
        self._dependency_windows = []
        
        self._f = f
        self._args = list(args or [])
        # Normalize window args before use
        self._args_windows = list(args_windows) if args_windows is not None else [None] * len(self._args)
        self._kwargs = kwargs or {}
        self._kwargs_windows = dict(kwargs_windows) if kwargs_windows is not None else {}
        for i in range(len(self._args)):
            if isinstance(self._args[i], InfiniteTensor):
                assert self._args_windows[i] is not None, f"Argument window must be provided for infinite tensors (arg {i})"
                dependent_dims = sum(1 for dim in self.shape if dim is None)
                self._args[i]._dependency_windows.append((self._args_windows[i], dependent_dims))
            else:
                assert self._args_windows[i] is None, f"Argument window must not be provided for non-infinite tensors (arg {i})"
        for kwarg in self._kwargs:
            if isinstance(self._kwargs[kwarg], InfiniteTensor):
                assert self._kwargs_windows.get(kwarg) is not None, f"Argument window must be provided for infinite tensors (kwarg {kwarg})"
                dependent_dims = sum(1 for dim in self.shape if dim is None)
                self._kwargs[kwarg]._dependency_windows.append((self._kwargs_windows[kwarg], dependent_dims))
            else:
                assert self._kwargs_windows.get(kwarg) is None, f"Argument window must not be provided for non-infinite tensors (kwarg {kwarg})"
        self._output_window = output_window
        
        # Calculate tile shape
        self._tile_shape = []
        i = 0
        for dim in self._shape:
            if dim is None:
                self._tile_shape.append(self._chunk_size[i])
                i += 1
            else:
                self._tile_shape.append(dim)
        self._tile_shape = tuple(self._tile_shape)
        _elem_size = torch.empty((), dtype=self._dtype).element_size()
        self._tile_bytes = int(np.prod(list(self._tile_shape))) * _elem_size
        
        self._marked_for_cleanup = False
        self._fully_cleaned = False
    
    @property
    def shape(self) -> tuple[int|None, ...]:
        return self._shape
    
    @property
    def uuid(self) -> UUID:
        return self._uuid

    def _calculate_indexed_shape(self, indices: tuple[slice, ...]) -> list[int]:
        """Calculate the shape of the result when indexing with given indices.
        
        This helper method eliminates code duplication between __getitem__, __setitem__,
        and _add_op methods.
        
        Args:
            indices: Normalized indices tuple
            
        Returns:
            List of integers representing the shape of the indexed region
        """
        output_shape = []
        for idx, shape_dim in zip(indices, self._shape):
            if isinstance(idx, slice):
                size = (idx.stop - idx.start - 1) // (idx.step or 1) + 1
                output_shape.append(size)
            else:
                output_shape.append(1)
        return output_shape

    def _validate_and_prepare_value(self, indices: tuple[slice, ...], value: torch.Tensor, 
                                   collapse_dims: list[int]) -> torch.Tensor:
        """Validate value shape and prepare it for assignment.
        
        Args:
            indices: Normalized indices tuple
            value: Value to validate and prepare
            collapse_dims: Dimensions that should be collapsed
            
        Returns:
            Prepared tensor with correct shape and dtype
            
        Raises:
            ValueError: If value shape doesn't match expected shape
        """
        expected_shape = self._calculate_indexed_shape(indices)
        expected_output_shape = tuple(s for i, s in enumerate(expected_shape) if i not in collapse_dims)
        
        value = torch.as_tensor(value, dtype=self._dtype)
        if value.shape != expected_output_shape:
            raise ShapeMismatchError(SHAPE_MISMATCH_ERROR_MSG.format(actual=value.shape, expected=expected_output_shape))
            
        # Expand value to match indexed shape if needed
        if collapse_dims:
            value = value.reshape(expected_shape)
            
        return value

    def _get_tile_key(self, tile_index: tuple[int, ...]) -> tuple[UUID, tuple[int, ...]]:
        """Generate a unique key for storing/retrieving a tile.
        
        Combines the tensor's UUID with the tile index to create a globally unique
        identifier for this specific tile.
        
        Args:
            tile_index: N-dimensional index of the tile in tile space
            
        Returns:
            Tuple containing (tensor_uuid, tile_index) for unique identification
        """
        return (self._uuid, tile_index)

    def _pixel_slices_to_tile_ranges(self, slices: tuple[slice, ...]) -> tuple[range, ...]:
        """Convert pixel-space slices to tile-space ranges.
        
        Takes pixel coordinates and determines which tiles contain those pixels.
        Only processes infinite dimensions (where shape[i] is None).
        
        Args:
            slices: Tuple of slices in pixel space
            
        Returns:
            Tuple of ranges indicating which tiles are needed for each infinite dimension
            
        Example:
            If chunk_size is 512 and we want pixels [100:1500], this returns
            range(0, 3) since we need tiles 0, 1, and 2 to cover that pixel range.
        """
        tile_ranges = []
        i = 0
        for j, pixel_range in enumerate(slices):
            if self._shape[j] is None:
                tile_ranges.append(range(pixel_range.start // self._chunk_size[i],
                                        (pixel_range.stop - 1) // self._chunk_size[i] + 1))
                i += 1
        return tuple(tile_ranges)
        
    def _intersect_slices(self, slices: tuple[slice, ...], tile_idx: tuple[int, ...]) -> tuple[slice, ...]:
        """Calculate the intersection of pixel-space slices with specific tile boundaries.

        This is a crucial function for determining which portion of a requested slice
        actually exists within a specific tile. It handles step sizes correctly to
        ensure only valid positions are included.

        Args:
            slices: Tuple of slices in pixel space to intersect with tile boundaries
            tile_idx: Tuple of integers specifying the tile index in tile space
            
        Returns:
            Tuple of slices representing the intersection of the input slices with 
            the tile boundaries, preserving step alignment
            
        Internal Logic:
            - For infinite dimensions: clips slice to tile boundaries while respecting step
            - For finite dimensions: passes slice through unchanged
            - Ensures step alignment by adjusting start/stop to valid step positions
        """
        infinite_dim = 0
        output = []
        for i, s in enumerate(slices):
            if self._shape[i] is None:
                # Calculate tile boundaries
                tile_start = tile_idx[infinite_dim] * self._chunk_size[infinite_dim]
                tile_end = (tile_idx[infinite_dim] + 1) * self._chunk_size[infinite_dim]
                
                # Adjust start to align with step
                start = max(s.start, tile_start)
                if s.step > 1:
                    # Adjust start to next valid step position after tile boundary
                    offset = (start - s.start) % s.step
                    if offset:
                        start += (s.step - offset)
                
                # Adjust end to align with step
                stop = min(s.stop, tile_end)
                if s.step > 1:
                    # Adjust stop to last valid step position before tile boundary
                    offset = (stop - 1 - s.start) % s.step
                    if offset:
                        stop -= offset
                
                output.append(slice(start, stop, s.step))
                infinite_dim += 1
            else:
                output.append(s)
        return tuple(output)
        
    def _translate_slices(self, slices: tuple[slice, ...], tile_idx: tuple[int, ...]) -> tuple[int, ...]:
        """Translate pixel-space slices to tile-local coordinates.
        
        Converts global pixel coordinates to coordinates within a specific tile.
        This is essential for indexing into tile.values tensors.
        
        Args:
            slices: Tuple of slices in global pixel space
            tile_idx: Tuple specifying which tile these slices refer to
            
        Returns:
            Tuple of slices with coordinates translated to be relative to tile origin
            
        Example:
            If tile_idx=(1,) and chunk_size=(512,), then pixel slice [600:700]
            becomes tile-local slice [88:188] (600-512:700-512)
        """
        infinite_dim = 0
        output = []
        for i, s in enumerate(slices):
            if self._shape[i] is None:
                output.append(slice(s.start - tile_idx[infinite_dim] * self._chunk_size[infinite_dim],
                                    s.stop - tile_idx[infinite_dim] * self._chunk_size[infinite_dim],
                                    s.step))
                infinite_dim += 1
            else:
                output.append(s)
        return tuple(output)
        
    def __getitem__(self, indices: tuple[int|slice, ...]) -> torch.Tensor:
        """Get a slice of the tensor.
        
        Args:
            indices: Tuple of indices/slices to access. Can include integers, slices, and ellipsis.
                    For infinite dimensions, slices can extend beyond defined regions.
                    
        Returns:
            torch.Tensor: The requested slice of the tensor.
            
        Examples:
            # Get a single value
            tensor[0, 0, 0]
            
            # Get a slice
            tensor[0:10, 0:10, 0:10] 
            
            # Use ellipsis to fill in remaining dimensions
            tensor[0, ...] # Equivalent to tensor[0, :, :, :] when tensor is 4D
            
            # Mix integers and slices
            tensor[0, :, 0:10]
        """
        indices, collapse_dims = standardize_indices(self._shape, indices)
        tile_ranges = self._pixel_slices_to_tile_ranges(indices)
        
        logger.debug(f"Accessing tensor slice with indices: {indices}")
        self._apply_f_range(indices)
        
        # Calculate output shape
        output_shape = self._calculate_indexed_shape(indices)
        logger.debug(f"Calculated output shape: {output_shape}")
                
        # Create output tensor
        output_tensor = torch.empty(output_shape, dtype=self._dtype)
        for tile_index in itertools.product(*tile_ranges):
            tile = self._store.get(self._get_tile_key(tile_index))
            if tile is None:
                raise TileAccessError(TILE_DELETED_ERROR_MSG)
            intersected_indices = self._intersect_slices(indices, tile_index)
            tile_space_indices = self._translate_slices(intersected_indices, tile_index)
            output_indices = tuple(slice((s.start - n.start) // s.step, (s.stop - n.start + s.step - 1) // s.step)
                                   for s, n in zip(intersected_indices, indices))
            
            output_tensor[output_indices] = tile.values[tile_space_indices]
            
        if collapse_dims:
            target_shape = tuple(s for i, s in enumerate(output_shape) if i not in collapse_dims)
            return output_tensor.reshape(target_shape)
        return output_tensor
            
    def __setitem__(self, indices: tuple[int|slice, ...], value: torch.Tensor):
        """Set a slice of the tensor.
        
        Args:
            indices: Tuple of indices/slices to set. Can include integers, slices, and ellipsis.
                    For infinite dimensions, slices can extend beyond defined regions.
            value: Value to set at the specified indices. Must match the shape of the indexed region.
                  Will be automatically converted to a tensor with the correct dtype.
                  
        Raises:
            ValueError: If the value shape does not match the indexed shape.
            
        Examples:
            # Set a single value
            tensor[0, 0, 0] = 1.0
            
            # Set a slice
            tensor[0:10, 0:10, 0:10] = torch.ones(10, 10, 10)
            
            # Use ellipsis to fill in remaining dimensions 
            tensor[0, ...] = torch.ones(10, 10, 10) # When tensor is 4D
            
            # Mix integers and slices
            tensor[0, :, 0:10] = torch.ones(5, 10)
        """
        if self._dependency_windows or self._marked_for_cleanup:
            raise DependencyError(DEPENDENCY_ERROR_MSG)
        
        indices, collapse_dims = standardize_indices(self._shape, indices)
        tile_ranges = self._pixel_slices_to_tile_ranges(indices)
        
        self._apply_f_range(indices)
        
        # Validate and prepare value
        value = self._validate_and_prepare_value(indices, value, collapse_dims)
            
        # Set values in tiles
        for tile_index in itertools.product(*tile_ranges):
            tile = self._store.get(self._get_tile_key(tile_index))
            if tile is None:
                raise TileAccessError(TILE_DELETED_ERROR_MSG)
            intersected_indices = self._intersect_slices(indices, tile_index)
            tile_space_indices = self._translate_slices(intersected_indices, tile_index)
            value_indices = tuple(slice((s.start - n.start) // s.step, (s.stop - n.start - 1) // s.step + 1)
                                for s, n in zip(intersected_indices, indices))
            
            tile.values[tile_space_indices] = value[value_indices]
            self._store.set(self._get_tile_key(tile_index), tile)
            
    def _add_op(self, indices: tuple[int|slice, ...], value: torch.Tensor):
        """Optimized addition operation that accumulates values into tiles.
        
        This is the primary method for writing computed results into the tensor.
        Unlike __setitem__, this adds to existing values and can create tiles on-demand.
        It's used internally during function application to accumulate results.

        Args:
            indices: Tuple of indices/slices where to add the value
            value: Tensor value to add (will be converted to correct dtype)

        Raises:
            ValueError: If value shape doesn't match indexed shape or devices differ
            
        Internal Behavior:
            - Creates zero-initialized tiles if they don't exist
            - Accumulates values using += operation
            - Handles device validation (tensors must be on CPU)
            - Does not trigger function application (unlike __getitem__)
        """
        indices, collapse_dims = standardize_indices(self._shape, indices)
        tile_ranges = self._pixel_slices_to_tile_ranges(indices)
        
        #self._apply_f_range(indices)
        
        # Validate and prepare value
        value = self._validate_and_prepare_value(indices, value, collapse_dims)
            
        # Set values in tiles
        for tile_index in itertools.product(*tile_ranges):
            tile = self._store.get(self._get_tile_key(tile_index))
            if tile is None:
                tile = InfinityTensorTile(values=torch.zeros(self._tile_shape, dtype=self._dtype))
            intersected_indices = self._intersect_slices(indices, tile_index)
            tile_space_indices = self._translate_slices(intersected_indices, tile_index)
            value_indices = tuple(slice((s.start - n.start) // s.step, (s.stop - n.start - 1) // s.step + 1)
                                for s, n in zip(intersected_indices, indices))
            
            # Assert tile values and value are on same device
            if tile.values.device != value.device:
                raise ValueError(DEVICE_MISMATCH_ERROR_MSG.format(actual=value.device))
            tile.values[tile_space_indices] += value[value_indices]
            self._store.set(self._get_tile_key(tile_index), tile)

    def _apply_f_range(self, pixel_range: tuple[slice, ...]):
        """Apply the generating function to all windows intersecting a pixel range.
        
        This is the core orchestration method that determines which windows need
        to be processed to generate data for a requested pixel range, then applies
        the function to each of those windows.
        
        Args:
            pixel_range: Tuple of slices defining the pixel region that needs data
            
        Internal Process:
            1. Calculate which windows intersect the requested pixel range
            2. Apply the function to each window that hasn't been processed yet
            3. Function outputs are accumulated into tiles via _add_op
        """
        lowest_window_indexes = self._output_window.get_lowest_intersection(pixel_range)
        highest_window_indexes = self._output_window.get_highest_intersection(pixel_range)
        for idx in itertools.product(*[range(l, h+1) for l, h in zip(lowest_window_indexes, highest_window_indexes)]):
            self._apply_f(idx)

    def _apply_f(self, window_index: tuple[int, ...]) -> torch.Tensor:
        """Apply the generating function to a specific window.
        
        This method orchestrates the application of the user-provided function to
        generate data for a specific window. It handles dependency resolution,
        argument preparation, and result accumulation.
        
        Args:
            window_index: N-dimensional index of the window in window space
            
        Returns:
            None (results are stored in tiles via _add_op)
            
        Internal Process:
            1. Check if window is already processed (skip if so)
            2. Prepare arguments by slicing dependent tensors using their windows
            3. Call user function with window context and prepared arguments
            4. Validate output shape matches expected window shape
            5. Accumulate result into tiles using _add_op
            6. Mark dependencies as processed for cleanup tracking
        """
        if self._store.is_window_processed(window_index):
            logger.debug(f"Window {window_index} already processed, skipping")
            return
        
        logger.debug(f"Processing window {window_index}")
        self._store.mark_window_processed(window_index)
        
        args = []
        for i, arg_window in enumerate(self._args_windows):
            if arg_window is not None:
                arg_window = self._args_windows[i]
                args.append(self._args[i][arg_window.get_bounds(window_index)])
            else:
                args.append(self._args[i])
        kwargs = {}
        for kwarg in self._kwargs:
            if self._kwargs_windows[kwarg] is not None:
                window = self._kwargs_windows[kwarg]
                kwargs[kwarg] = self._kwargs[kwarg][window.get_bounds(window_index)]
            else:
                kwargs[kwarg] = self._kwargs[kwarg]
        output = self._f(window_index, *args, **kwargs)
        
        # Verify output shape matches the expected window shape
        expected_shape = self._output_window.window_size
        if tuple(output.shape) != tuple(expected_shape):
            raise ShapeMismatchError(OUTPUT_SHAPE_ERROR_MSG.format(actual=output.shape, expected=expected_shape))
        
        self._add_op(self._output_window.get_bounds(window_index), output)
        
        for i, arg in enumerate(self._args):
            if isinstance(arg, InfiniteTensor):
                arg._mark_dependency_processed(self._args_windows[i], window_index)
        for kwarg in self._kwargs:
            if isinstance(self._kwargs[kwarg], InfiniteTensor):
                self._kwargs[kwarg]._mark_dependency_processed(self._kwargs_windows[kwarg], window_index)

    def _mark_dependency_processed(self, window, window_index: tuple[int, ...]):
        """Mark a dependency window as processed and handle cleanup.
        
        This is called when a dependent tensor processes a window that references
        this tensor. It increments the processing counter for affected tiles and
        triggers cleanup if tiles are no longer needed.
        
        Args:
            window: The TensorWindow that was processed in the dependent tensor
            window_index: Index of the processed window in the dependent tensor
            
        Internal Process:
            1. Determine which tiles are affected by the processed window
            2. Increment dependency_windows_processed counter for each tile
            3. If tensor is marked for cleanup, delete tiles that are no longer needed
            4. Attempt full cleanup if all dependencies are satisfied
        """
        tile_ranges = self._pixel_slices_to_tile_ranges(window.get_bounds(window_index))
        for tile_index in itertools.product(*tile_ranges):
            tile = self._store.get(self._get_tile_key(tile_index))
            tile.dependency_windows_processed += 1
            self._store.set(self._get_tile_key(tile_index), tile)
            if self._marked_for_cleanup and not self._is_tile_needed(tile_index):
                self._store.delete(self._get_tile_key(tile_index))
        self._full_cleanup()
    
    def _is_tile_needed(self, tile_index: tuple[int, ...]) -> bool:
        """Determine if a tile is still needed by dependency tracking.
        
        This is the core logic for automatic memory management. It calculates
        how many dependency windows should reference this tile and compares
        that to how many have actually been processed.
        
        Args:
            tile_index: Index of the tile to check
            
        Returns:
            True if the tile is still needed, False if it can be safely deleted
            
        Internal Logic:
            1. Convert tile bounds to pixel space
            2. For each dependency window, calculate how many windows overlap this tile
            3. Handle special case of infinite dependencies (when dependent dimensions
               exceed mapped dimensions)
            4. Compare total expected windows to processed windows count
            5. Return True if any windows are still pending
        """
        tile_slices = self._get_tile_bounds(tile_index)
        
        total_windows = 0
        for dependency_window, dependent_dims in self._dependency_windows:
            window_range = dependency_window.pixel_range_to_window_range(tile_slices)
            if window_range is None:
                continue
            
            # This tile is needed by an infinite number of windows.
            # This can happen if a dependent tensor dimension is not mapped to any dimension in the current tensor.
            if dependency_window.dimension_map and dependent_dims > len(set(x for x in dependency_window.dimension_map if x is not None)):
                return True
            
            num_windows = 1
            for r in window_range:
                num_windows *= len(range(r.start, r.stop))
            total_windows += num_windows
        
        tile = self._store.get(self._get_tile_key(tile_index))
        if tile is None:
            return False # Already deleted
        return total_windows != tile.dependency_windows_processed

    def _get_tile_bounds(self, tile_index: tuple[int, ...]) -> tuple[slice, ...]:
        """Calculate the pixel-space boundaries of a tile.
        
        Converts a tile index in tile space to the corresponding pixel coordinates
        that the tile covers. Essential for dependency tracking and tile management.
        
        Args:
            tile_index: N-dimensional index of the tile in tile space
            
        Returns:
            Tuple of slices defining the pixel boundaries covered by this tile
            
        Example:
            For tile_index=(1, 2) with chunk_size=(512, 512), returns
            (slice(512, 1024), slice(1024, 1536)) for the pixel bounds
        """
        tile_slices = []
        infinite_dim = 0
        for i, dim in enumerate(self._shape):
            if dim is None:
                start = tile_index[infinite_dim] * self._chunk_size[infinite_dim]
                stop = (tile_index[infinite_dim] + 1) * self._chunk_size[infinite_dim]
                tile_slices.append(slice(start, stop))
                infinite_dim += 1
            else:
                tile_slices.append(slice(0, dim))
        return tuple(tile_slices)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.mark_for_cleanup()
        return False  # Don't suppress any exceptions

    def __del__(self):
        self.mark_for_cleanup()
                
    def mark_for_cleanup(self):
        """Mark this tensor for cleanup and begin aggressive memory management.
        
        This initiates the cleanup process by marking the tensor for deletion
        and immediately removing any tiles that are no longer needed. Called
        automatically during garbage collection (__del__).
        
        Process:
            1. Mark tensor as ready for cleanup
            2. Immediately delete any tiles that have no pending dependencies
            3. Trigger full cleanup if possible
        """
        if not self._marked_for_cleanup:
            for uuid, tile_index in self._store.keys():
                if not self._is_tile_needed(tile_index):
                    self._store.delete(self._get_tile_key(tile_index))
            self._full_cleanup()
            self._marked_for_cleanup = True
            
    def _full_cleanup(self):
        """Perform complete cleanup when tensor is no longer needed.
        
        This is the final cleanup phase that occurs when:
        1. Tensor is marked for cleanup
        2. All tiles have been processed and deleted
        3. No other tensors depend on this one
        
        Actions:
            - Clear the tile store completely
            - Remove this tensor from its dependencies' reference lists
            - Recursively trigger cleanup in dependency tensors
            - Mark tensor as fully cleaned to prevent duplicate cleanup
        """
        if self._marked_for_cleanup and len(self._store.keys()) == 0 and len(self._dependency_windows) == 0 and not self._fully_cleaned:
            self._fully_cleaned = True
            self._store.clear()
            for arg, arg_window in zip(self._args, self._args_windows):
                if isinstance(arg, InfiniteTensor):
                    dependent_dims = sum(1 for dim in self.shape if dim is None)
                    arg._dependency_windows.remove((arg_window, dependent_dims))
                    arg._full_cleanup()
            for kwarg in self._kwargs:
                arg = self._kwargs[kwarg]
                if isinstance(arg, InfiniteTensor):
                    window = self._kwargs_windows[kwarg]
                    dependent_dims = sum(1 for dim in self.shape if dim is None)
                    arg._dependency_windows.remove((window, dependent_dims))
                    arg._full_cleanup()
        
