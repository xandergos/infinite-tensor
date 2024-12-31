from dataclasses import dataclass, field
from functools import lru_cache
import gc
import math
import os
import random
import tempfile
import time
from typing import Any, Callable
import weakref
import numpy as np
import torch
import h5py
from collections import OrderedDict
import itertools
import matplotlib.pyplot as plt
import uuid
from uuid import UUID

# infinite pixel space - Pixel space but only infinite dimensions included
# finite pixel space - Pixel space with all dimensions included
# window space - Each point is a window of pixels
# tile space - Each point is a tile of pixels

# max cache size in bytes
_max_cache_size = 10**9
_global_cache_size = 0
# _global_cache is a dictionary mapping keys (uuid, tile_index) to (weakref.ref(tensor), tile)
_global_cache = OrderedDict()

def set_max_cache_size(size: int):
    global _max_cache_size
    _max_cache_size = size
    while _global_cache_size > _max_cache_size:
        _global_cache.popitem(last=False)

@dataclass
class InfinityTensorTile:
    values: torch.Tensor
    modified: bool = False
    operation_counter = 0  # Number of operations that are waiting to be applied
    counted_operations: set = field(default_factory=set)  # Set of operations that have been counted in operation_counter
    processed_windows: set = field(default_factory=set)  # Set of window indices. Used only in result tensors
    value_weights: torch.Tensor = None

def _normalize_slice(idx: int|slice, shape_dim: int|None):
    """Normalizes a slice so that stop is one greater than the last element. Assumes start/stop is not None."""
    if isinstance(idx, int):
        return slice(idx, idx + 1, 1)
    else:
        start, stop = idx.start, idx.stop
        if start is None:
            assert shape_dim is not None, "Slice must have start and stop for infinite dimensions"
            start = 0
        if stop is None:
            assert shape_dim is not None, "Slice must have start and stop for infinite dimensions"
            stop = shape_dim
        step = idx.step if idx.step is not None else 1
        return slice(start, start + ((stop - start - 1) // step) * step + 1, step)

class InfiniteTensor:
    def __init__(self, 
                 shape: tuple[int|None, ...],
                 chunk_size: int|tuple[int, ...] = 512, 
                 default_value = 0,
                 dtype: torch.dtype = torch.float32,
                 tile_init_fn: Callable = None):
        """Initialize an InfiniteTensor.

        An InfiniteTensor represents a theoretically infinite tensor that is processed in chunks.
        Operations can be performed on the tensor in a sliding window manner without loading
        the entire tensor into memory.

        Args:
            shape: Shape of the tensor. Use None to indicate that the tensor is infinite in that dimension. For example (3, 10, None, None) 
                   indicates a 4-dimensional tensor with the first two dimensions of size 3 and 10, and the last two dimensions being infinite.
                   This might be used to represent a batch of 3 images, with 10 channels and an infinite width and height.
            chunk_size: Size of each chunk. Can be an integer for uniform chunk size, or tuple of integers to specify a different chunk size for each dimension.
                        The number of 'None' values in 'shape' must match the number of dimensions in 'chunk_size' if it is a tuple.
            cache_size: Maximum number of tiles to keep in memory at once.
            default_value: Value to initialize tensor elements with, when the tensor is undefined (default: 0)
            dtype: PyTorch data type for the tensor (default: torch.float32)
            tile_init_fn: Function to initialize tiles with. If None, tiles are initialized with the default value.
                The function takes the tile coordinates as a tuple of integers, the desired size of the tile as a tuple, and returns a tensor of the appropriate size.
                Example: lambda tile_index, tile_shape: torch.randn(tile_shape)
        """
        self._shape = shape
        if isinstance(chunk_size, int):
            self._chunk_size = (chunk_size,) * sum(1 for dim in shape if dim is None)
        else:
            self._chunk_size = chunk_size
        self._default_value = default_value
        self._dtype = dtype
        self._tile_init_fn = tile_init_fn
        self._uuid = uuid.uuid4()
        self._operators = []
            
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
        self._tile_bytes = np.prod(list(self._tile_shape)) * self._dtype.itemsize
        
        # Open temporary file
        self._dir_path = tempfile.TemporaryDirectory(delete=True)
        self._file_path = os.path.join(self._dir_path.name, f'tensor.h5')
        self._h5file = h5py.File(self._file_path, 'w')
        self._cleanup_done = False
    
    @property
    def uuid(self) -> UUID:
        return self._uuid
    
    def _get_tile(self, *indices: int) -> InfinityTensorTile:
        """Get a tile of the tensor at the given indices.
        
        Args:
            index: Tuple of indices specifying the tile location
            
        Returns:
            InfinityTensorTile: The requested tile
        """
        global _global_cache_size
        
        dataset_key = self._get_dataset_key(indices)
        
        # Check if tile is in cache
        cache_key = self._get_cache_key(indices)
        if cache_key in _global_cache:
            # Move to end (most recently used)
            _global_cache.move_to_end(cache_key)
            cached_tensor, cached_tile = _global_cache[cache_key]
            assert cached_tensor() is self
            return cached_tile
        
        # If cache is full, remove least recently used tile
        future_cache_size = _global_cache_size + self._tile_bytes
        if future_cache_size >= _max_cache_size:
            oldest_key, (oldest_tensor, oldest_tile) = _global_cache.popitem(last=False)
            oldest_uuid, oldest_index = oldest_key
            
            if oldest_tensor is not None and oldest_tile.modified:
                self._save_tile(oldest_tile, oldest_index)
        
        # Try to load tile from HDF5
        if dataset_key in self._h5file:
            values = torch.from_numpy(self._h5file[dataset_key][:])
            tile = InfinityTensorTile(values=values)
        else:
            # Create new tile with default values
            if self._tile_init_fn is not None:
                values = self._tile_init_fn(indices, self._tile_shape)
                tile = InfinityTensorTile(values=values, modified=True)
            else:
                values = torch.full(self._tile_shape, self._default_value, dtype=self._dtype)
                tile = InfinityTensorTile(values=values, modified=False)  
    
        # Add to cache
        assert isinstance(cache_key[1], tuple) and isinstance(cache_key[1][0], int)
        _global_cache[cache_key] = (weakref.ref(self), tile)
        _global_cache_size += self._tile_bytes
        
        return tile

    def _get_cache_key(self, tile_index: tuple[int, ...]) -> tuple[UUID, tuple[int, ...]]:
        return (self._uuid, tile_index)
    
    def _get_dataset_key(self, tile_index: tuple[int, ...]) -> str:
        return ','.join(str(i) for i in tile_index)
    
    def _save_tile(self, tile: InfinityTensorTile, tile_index: tuple[int, ...]):
        dataset_key = self._get_dataset_key(tile_index)
        if dataset_key not in self._h5file:
            self._h5file.create_dataset(dataset_key, data=tile.values.numpy())
            dataset = self._h5file[dataset_key]
        else:
            dataset = self._h5file[dataset_key]
            dataset[:] = tile.values.numpy()
        dataset.attrs['operation_counter'] = tile.operation_counter
        dataset.attrs['counted_operations'] = '$'.join(str(t.uuid) for t in tile.counted_operations)
        dataset.attrs['processed_windows'] = np.array([list(w) for w in tile.processed_windows], dtype=int)
            
    def _remove_from_cache(self, tile_index: tuple[int, ...]):
        global _global_cache_size
        cache_key = self._get_cache_key(tile_index)
        if cache_key in _global_cache:
            # Remove from cache
            cached_tensor, cached_tile = _global_cache.pop(cache_key)
            assert cached_tensor() is self
            _global_cache_size -= self._tile_bytes
            
            # Update file if tile has been modified
            if cached_tile.modified:
                self._save_tile(cached_tile, tile_index)
    
    def clean_cache(self):
        for cache_key in list(_global_cache.keys()):
            if cache_key[0] == self._uuid:
                self._remove_from_cache(cache_key[1])

    def _pixel_slices_to_tile_ranges(self, slices: tuple[slice, ...]) -> tuple[range, ...]:
        tile_ranges = []
        for i, pixel_range in enumerate(slices):
            tile_ranges.append(range(pixel_range.start // self._chunk_size[i],
                                    (pixel_range.stop - 1) // self._chunk_size[i] + 1))
        return tuple(tile_ranges)
        
    def _intersect_slices(self, slices: tuple[slice, ...], tile_idx: tuple[int, ...]) -> tuple[slice, ...]:
        """Returns the intersection of the given slices with the tile boundaries.

        Args:
            slices: Tuple of slices to intersect with tile boundaries.
            tile_idx: Tuple of integers specifying the tile index.
        Returns:
            tuple[slice, ...]: Tuple of slices, each representing the intersection of the corresponding input slice with the tile boundaries.
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
        
    def _to_tile_space(self, slices: tuple[slice, ...], tile_index: tuple[int, ...]) -> tuple[int, ...]:
        infinite_dim = 0
        output = []
        for i, s in enumerate(slices):
            if self._shape[i] is None:
                output.append(slice(s.start - tile_index[infinite_dim] * self._chunk_size[infinite_dim],
                                    s.stop - tile_index[infinite_dim] * self._chunk_size[infinite_dim],
                                    s.step))
                infinite_dim += 1
            else:
                output.append(s)
        return tuple(output)
        
    def _get_slice_size(self, slice: slice, shape_dim: int|None) -> int:
        if shape_dim is None:
            return (slice.stop - slice.start) // slice.step
        else:
            return shape_dim
        
    def _standardize_indices(self, indices: tuple[int|slice, ...]) -> tuple[slice, ...]:
        """
        Converts indices to a normalized form, with all ellipses expanded and slices normalized.
        Also returns a list of dimensions that should be collapsed because integers were passed.
        """
        if isinstance(indices, list):
            indices = tuple(indices)
        if not isinstance(indices, tuple):
            indices = (indices,)
            
        # Handle ellipsis by expanding it to the appropriate number of full slices
        if Ellipsis in indices:
            assert indices.count(Ellipsis) == 1, "Only one ellipsis is allowed"
            ellipsis_idx = indices.index(Ellipsis)
            n_missing = len(self._shape) - len(indices) + 1
            expanded_indices = (
                indices[:ellipsis_idx] + 
                (slice(None),) * n_missing +
                indices[ellipsis_idx + 1:]
            )
        else:
            expanded_indices = indices
            
        # Pad with full slices if needed
        if len(expanded_indices) < len(self._shape):
            expanded_indices = expanded_indices + (slice(None),) * (len(self._shape) - len(expanded_indices))
            
        # Validate indices length matches shape
        if len(expanded_indices) != len(self._shape):
            raise IndexError(f"Too many indices for infinite tensor of dimension {len(self._shape)}")
        
        collapse_dims = [i for i, idx in enumerate(expanded_indices) if isinstance(idx, int)]
        return [_normalize_slice(idx, shape_dim) for idx, shape_dim in zip(expanded_indices, self._shape)], collapse_dims
        
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
        indices, collapse_dims = self._standardize_indices(indices)
        inf_indices = [idx for i, idx in enumerate(indices) if self._shape[i] is None]
        tile_ranges = self._pixel_slices_to_tile_ranges(inf_indices)
        
        # Calculate output shape
        output_shape = []
        for i, (idx, shape_dim) in enumerate(zip(indices, self._shape)):
            if isinstance(idx, slice):
                size = (idx.stop - idx.start - 1) // (idx.step or 1) + 1
                output_shape.append(size)
            else:
                output_shape.append(1)
                
        # Create output tensor
        output_tensor = torch.empty(output_shape, dtype=self._dtype)
        for tile_index in itertools.product(*tile_ranges):
            tile = self._get_tile(*tile_index)
            intersected_indices = self._intersect_slices(indices, tile_index)
            tile_space_indices = self._to_tile_space(intersected_indices, tile_index)
            output_indices = tuple(slice((s.start - n.start) // s.step, (s.stop - n.start + s.step - 1) // s.step)
                                   for s, n in zip(intersected_indices, indices))
            
            if tile.value_weights is not None:
                output_tensor[output_indices] = tile.values[tile_space_indices] / tile.value_weights[tile_space_indices]
            else:
                output_tensor[output_indices] = tile.values[tile_space_indices]
            
        return torch.squeeze(output_tensor, collapse_dims)
            
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
        indices, collapse_dims = self._standardize_indices(indices)
        inf_indices = [idx for i, idx in enumerate(indices) if self._shape[i] is None]
        tile_ranges = self._pixel_slices_to_tile_ranges(inf_indices)
        
        # Validate value shape matches indexed shape
        expected_shape = []
        for i, (idx, shape_dim) in enumerate(zip(indices, self._shape)):
            if isinstance(idx, slice):
                size = (idx.stop - idx.start - 1) // (idx.step or 1) + 1
                expected_shape.append(size)
            else:
                expected_shape.append(1)
                
        value = torch.as_tensor(value, dtype=self._dtype)
        if value.shape != tuple(s for i,s in enumerate(expected_shape) if i not in collapse_dims):
            raise ValueError(f"Value shape {value.shape} does not match indexed shape {tuple(s for i,s in enumerate(expected_shape) if i not in collapse_dims)}")
            
        # Expand value to match indexed shape if needed
        if collapse_dims:
            expanded_shape = list(expected_shape)
            value = value.reshape(expanded_shape)
            
        # Set values in tiles
        for tile_index in itertools.product(*tile_ranges):
            tile = self._get_tile(*tile_index)
            intersected_indices = self._intersect_slices(indices, tile_index)
            tile_space_indices = self._to_tile_space(intersected_indices, tile_index)
            value_indices = tuple(slice((s.start - n.start) // s.step, (s.stop - n.start - 1) // s.step + 1)
                                for s, n in zip(intersected_indices, indices))
            
            tile.values[tile_space_indices] = value[value_indices]
            tile.modified = True
            
    def add(self, indices: tuple[int|slice, ...], value: torch.Tensor):
        """Shortcut for self[indices] = self[indices] + value.

        Args:
            indices (tuple[int | slice, ...]): Indices to add the value to.
            value (torch.Tensor): Value to add to the tensor.

        Raises:
            ValueError: If the value shape does not match the indexed shape.
        """
        indices, collapse_dims = self._standardize_indices(indices)
        inf_indices = [idx for i, idx in enumerate(indices) if self._shape[i] is None]
        tile_ranges = self._pixel_slices_to_tile_ranges(inf_indices)
        
        # Validate value shape matches indexed shape
        expected_shape = []
        for i, (idx, shape_dim) in enumerate(zip(indices, self._shape)):
            if isinstance(idx, slice):
                size = (idx.stop - idx.start - 1) // (idx.step or 1) + 1
                expected_shape.append(size)
            else:
                expected_shape.append(1)
                
        value = torch.as_tensor(value, dtype=self._dtype)
        if value.shape != tuple(s for i,s in enumerate(expected_shape) if i not in collapse_dims):
            raise ValueError(f"Value shape {value.shape} does not match indexed shape {tuple(s for i,s in enumerate(expected_shape) if i not in collapse_dims)}")
            
        # Expand value to match indexed shape if needed
        if collapse_dims:
            expanded_shape = list(expected_shape)
            value = value.reshape(expanded_shape)
            
        # Set values in tiles
        for tile_index in itertools.product(*tile_ranges):
            tile = self._get_tile(*tile_index)
            intersected_indices = self._intersect_slices(indices, tile_index)
            tile_space_indices = self._to_tile_space(intersected_indices, tile_index)
            value_indices = tuple(slice((s.start - n.start) // s.step, (s.stop - n.start - 1) // s.step + 1)
                                for s, n in zip(intersected_indices, indices))
            
            tile.values[tile_space_indices] += value[value_indices]
            tile.modified = True
    
    def queue_prepare_range(self, indices: tuple[slice, ...]):
        """Prepares the provided indices for sampling by processing all windows in the range.
        Indices are expected to be infinite pixel ranges."""
        pass

    def infinite_to_finite_indices(self, indices: tuple[slice, ...]) -> tuple[slice, ...]:
        """Converts infinite pixel space indices to finite pixel space indices."""
        out = []
        i = 0
        for s in self._shape:
            if s is None:
                out.append(indices[i])
                i += 1
            else:
                out.append(slice(None))
        return tuple(out)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False  # Don't suppress any exceptions

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        """Clean up resources used by this tensor."""
        if not self._cleanup_done:
            self.clean_cache()  # Save any modified tiles
            if hasattr(self, '_h5file'):
                self._h5file.close()
            if hasattr(self, '_file_path') and os.path.exists(self._file_path):
                try:
                    os.remove(self._file_path)
                except (OSError, PermissionError):
                    pass  # Best effort deletion
            self._cleanup_done = True
            
    @classmethod
    def apply(cls, f, args, kwargs, args_windows, kwargs_windows):
        pass

class TensorWindow:
    def __init__(self, tensor: InfiniteTensor, 
                 window_size: tuple[int, ...], 
                 window_stride: tuple[int, ...], 
                 window_offset: tuple[int, ...] = 0):
        """A sliding window that can be used to apply a function to an infinite tensor.
        
        Args:
            tensor (InfiniteTensor): The infinite tensor to use as input to the function.
            window_size (tuple[int, ...]): The size of the window for each dimension.
            window_stride (tuple[int, ...]): The stride between windows for each dimension.
            window_offset (tuple[int, ...], optional): The offset of the window for each dimension. 
                Defaults to 0, where the top-left corner of the window at index (0, 0, ...) is at the origin.
        """
        self.tensor = tensor
        self.window_size = window_size if isinstance(window_size, tuple) else (window_size,) * len(tensor._tile_shape)
        self.window_stride = window_stride if isinstance(window_stride, tuple) else (window_stride,) * len(tensor._tile_shape)
        self.window_offset = window_offset if isinstance(window_offset, tuple) else (window_offset,) * len(tensor._tile_shape)
        
    def get_lowest_intersection(self, point: tuple[int, ...]) -> tuple[int, ...]:
        """Returns the lowest window index that intersects with the given point.
        If there is no window that intersects, returns the next window index."""
        # Calculate window indices that would contain this point
        window_indices = []
        for p, w, s, o in zip(point, self.window_size, self.window_stride, self.window_offset):
            # Find lowest window index where window end > point
            # Window end = index * stride + offset + size
            # Solve for index: p < i * s + o + w
            # (p - o - w) / s < i
            # Calculate ceiling division without float math:
            # For positive numbers: (a + b - 1) // b gives ceiling division
            # For negative numbers: -((-a) // b)
            numerator = p - o - w + 1
            if numerator >= 0:
                idx = (numerator + s - 1) // s
            else:
                idx = -(-numerator // s)
            window_indices.append(idx)
            
        return tuple(window_indices)
        
    def get_highest_intersection(self, point: int|slice) -> tuple[int, ...]:
        """Returns the highest window index that intersects with the given point."""
        # Calculate window indices that would contain this point
        window_indices = []
        for p, w, s, o in zip(point, self.window_size, self.window_stride, self.window_offset):
            if isinstance(p, slice):
                # For slices, use stop-1 since that's the last point included
                p = p.stop - 1
                
            # Find highest window index where window start <= point
            # Window start = index * stride + offset
            # Solve for index: p >= i * s + o
            # i <= (p - o) / s
            # Calculate floor division
            idx = (p - o) // s
            window_indices.append(idx)
            
        return tuple(window_indices)
    
    def pixel_range_to_window_range(self, *inf_pixel_ranges: slice) -> tuple[slice, ...]:
        """Returns the window ranges that intersect with the given pixel ranges. Returns None if there is no intersection.
        Pixel ranges are expected to be infinite pixel ranges."""
        lowest_intersection = self.get_lowest_intersection((p.start for p in inf_pixel_ranges))
        highest_intersection = self.get_highest_intersection((p.stop - 1 for p in inf_pixel_ranges))
        if any(l >= h for l, h in zip(lowest_intersection, highest_intersection)):
            return None  # This means one of the pixel ranges is between two windows
        return tuple(slice(l, h + 1) for l, h in zip(lowest_intersection, highest_intersection))
    
    def get_bounds(self, *window_slices: slice|int) -> tuple[slice, ...]:
        """Returns the bounds of the given window slices in infinite pixel space."""
        window_slices = [_normalize_slice(w, None) for w in window_slices]
        return tuple(slice(w.start * stride + offset, (w.stop - 1) * stride + size + offset)
                     for w, stride, size, offset in zip(window_slices, self.window_stride, self.window_size, self.window_offset))

class InfiniteTensorResult(InfiniteTensor):
    """An infinite tensor that represents the result of an operation on other infinite tensors.
    
    This class represents a tensor that is computed lazily as the result of applying
    some function to one or more input tensors. The computation is performed tile-by-tile
    as needed, rather than computing the entire result at once.

    The function is applied with a sliding window approach, where each output tile is
    computed from the corresponding region of the input tensor(s). The input and output
    regions can have different sizes, controlled by the input_size/stride and 
    output_size/stride parameters.

    Args:
        f: The function to apply to compute the result
        args: Positional arguments to pass to the function
        kwargs: Keyword arguments to pass to the function
        input_size: Size of input window for each dimension
        input_stride: Stride between input windows for each dimension
        output_size: Size of output window for each dimension
        output_stride: Stride between output windows for each dimension
        shape: Shape of the result tensor
        chunk_size: Size of chunks for processing infinite dimensions
    """
    def __init__(self, f, args, kwargs,
                 window_size: int|tuple[int, ...],
                 window_stride: int|tuple[int, ...],
                 shape: tuple[int|None, ...],
                 window_offset: int|tuple[int, ...] = 0,
                 chunk_size: int|tuple[int, ...] = 512,
                 output_weights: torch.Tensor|None = None):
        """A tensor that represents the result of an operation on other tensors.
        
        This class represents a tensor that is computed lazily as the result of applying
        some function to one or more input tensors. The computation is performed tile-by-tile
        as needed, rather than computing the entire result at once.

        The function is applied with a sliding window approach, where each output tile is
        computed from the corresponding region of the input tensor(s). The input and output
        regions can have different sizes, controlled by the input_size/stride and 
        output_size/stride parameters.

        Args:
            f: The function to apply to compute the result
            args: Positional arguments to pass to the function
            kwargs: Keyword arguments to pass to the function
            input_size: Size of input window for each dimension
            input_stride: Stride between input windows for each dimension
            output_size: Size of output window for each dimension
            output_stride: Stride between output windows for each dimension
            shape: Shape of the result tensor. The shape should have the same number of infinite dimensions as the input tensors.
            chunk_size: Size of chunks for processing infinite dimensions
            output_weights: Weights to apply to the output values. If None, the output values are just summed.
                Otherwise, the weighted mean of all outputs is computed. For example, a value of 1 means all outputs are weighted equally,
                and each pixel will be the mean of all contributing sliding windows.
                The shape of the weights tensor should broadcast with the output tensor.
        """
        super().__init__(shape, chunk_size)
        self.f = f
        self.args = args
        self.kwargs = kwargs
        self._sliding_window = TensorWindow(None,
                                             window_size if isinstance(window_size, tuple) else (window_size,) * len(self._tile_shape),
                                             window_stride if isinstance(window_stride, tuple) else (window_stride,) * len(self._tile_shape),
                                             window_offset if isinstance(window_offset, tuple) else (window_offset,) * len(self._tile_shape))
        self.output_weights = output_weights
        for x in args:
            if isinstance(x, TensorWindow):
                assert x.tensor is not None, "SlidingWindow must be initialized with a tensor"
                self._operators.append(x.tensor)
        for v in kwargs.values():
            if isinstance(v, TensorWindow):
                assert v.tensor is not None, "SlidingWindow must be initialized with a tensor"
                self._operators.append(v.tensor)
    
    def _is_window_processed(self, window_index: tuple[int, ...]) -> bool:
        """Returns whether the given window has been processed."""
        tile_idx = [s.start // self._chunk_size[i] for i, s in enumerate(self._sliding_window.get_bounds(*window_index))]
        tile = self._get_tile(*tile_idx)
        return window_index in tile.processed_windows
    
    def _mark_window_processed(self, window_index: tuple[int, ...]):
        """Marks the given window as processed."""
        tile_idx = [s.start // self._chunk_size[i] for i, s in enumerate(self._sliding_window.get_bounds(*window_index))]
        tile = self._get_tile(*tile_idx)
        tile.processed_windows.add(window_index)
    
    def process_windows(self, window_slices: tuple[slice, ...]):
        window_slices = [range(s.start, s.stop) for s in window_slices]
        
        # Prepare input tensors
        for window_index in itertools.product(*window_slices):
            # Check if the window has already been processed
            if self._is_window_processed(window_index):
                continue
            
            # If the window has not been processed, prepare the input tensors
            for arg in self.args:
                if isinstance(arg, TensorWindow):
                    input_bounds = arg.get_bounds(*window_index)
                    arg.tensor.queue_prepare_range(input_bounds)
            for v in self.kwargs.values():
                if isinstance(v, TensorWindow):
                    input_bounds = v.get_bounds(*window_index)
                    v.tensor.queue_prepare_range(input_bounds)
                
        for window_index in itertools.product(*window_slices):
            if self._is_window_processed(window_index):
                continue
            
            output_bounds = self._sliding_window.get_bounds(*window_index)
            
            args = []
            kwargs = {}
            for arg in self.args:
                if isinstance(arg, TensorWindow):
                    input_bounds = self.infinite_to_finite_indices(arg.get_bounds(*window_index))
                    args.append(arg.tensor[input_bounds])
                else:
                    args.append(arg)
            for k, v in self.kwargs.items():
                if isinstance(v, TensorWindow):
                    input_bounds = self.infinite_to_finite_indices(v.get_bounds(*window_index))
                    kwargs[k] = v.tensor[input_bounds]
                else:
                    kwargs[k] = v
            
            # Apply function
            result = self.f(*args, **kwargs)
            try:
                self._weighted_add(self.infinite_to_finite_indices(output_bounds), result, self.output_weights)
            except ValueError as e:
                print(f"Operation function returned a value with the incorrect shape. Ensure the input and output window sizes match.")
                raise e
            
            self._mark_window_processed(window_index)
        
    def queue_prepare_range(self, indices: tuple[slice, ...]):
        """Indicates that the provided indices will be accessed soon.
        Indices are expected to be infinite pixel ranges.
        The windows may be processed immediately, or later in batches depending on strategy."""
        
        # Get tile ranges for infinite dimensions
        normalized_indices = [_normalize_slice(idx, None) for idx in indices]
        window_ranges = self._sliding_window.pixel_range_to_window_range(*[normalized_indices[i] for i, shape_val in enumerate(self._shape) if shape_val is None])
                
        if window_ranges is None:  # This means one of the indices is between two windows, don't need to do any calculations
            return
                
        # Process all windows
        self.process_windows(window_ranges)
    
    def __getitem__(self, indices: tuple[int|slice, ...]) -> torch.Tensor:
        # Get tile ranges for infinite dimensions
        normalized_indices = [_normalize_slice(idx, shape_dim) for idx, shape_dim in zip(indices, self._shape)]
        assert len(normalized_indices) == len(self._shape), "Slicing is missing dimensions"
        inf_indices = [idx for idx, shape_val in zip(normalized_indices, self._shape) if shape_val is None]
        window_ranges = self._sliding_window.pixel_range_to_window_range(*inf_indices)
                
        if window_ranges is None:  # This means one of the indices is between two windows, don't need to do any calculations
            return super().__getitem__(indices)
                
        # Process all windows
        self.process_windows(window_ranges)
        
        return super().__getitem__(indices)
    
    def __setitem__(self, indices: tuple[int|slice, ...], value: torch.Tensor):
        raise NotImplementedError("Setting values is only supported for root InfinityTensor.")
    
    def _weighted_add(self, indices: tuple[int|slice, ...], value: torch.Tensor, weights: torch.Tensor):
        """Shortcut for self[indices] = self[indices] + value.

        Args:
            indices (tuple[int | slice, ...]): Indices to add the value to.
            value (torch.Tensor): Value to add to the tensor.

        Raises:
            ValueError: If the value shape does not match the indexed shape.
        """
        indices, collapse_dims = self._standardize_indices(indices)
        inf_indices = [idx for i, idx in enumerate(indices) if self._shape[i] is None]
        tile_ranges = self._pixel_slices_to_tile_ranges(inf_indices)
        
        # Validate value shape matches indexed shape
        expected_shape = []
        for i, (idx, shape_dim) in enumerate(zip(indices, self._shape)):
            if isinstance(idx, slice):
                size = (idx.stop - idx.start - 1) // (idx.step or 1) + 1
                expected_shape.append(size)
            else:
                expected_shape.append(1)
                
        value = torch.as_tensor(value, dtype=self._dtype)
        if value.shape != tuple(s for i,s in enumerate(expected_shape) if i not in collapse_dims):
            raise ValueError(f"Value shape {value.shape} does not match indexed shape {tuple(s for i,s in enumerate(expected_shape) if i not in collapse_dims)}")
            
        # Expand value to match indexed shape if needed
        if collapse_dims:
            expanded_shape = list(expected_shape)
            value = value.reshape(expanded_shape)
            
        # Set values in tiles
        for tile_index in itertools.product(*tile_ranges):
            tile = self._get_tile(*tile_index)
            intersected_indices = self._intersect_slices(indices, tile_index)
            tile_space_indices = self._to_tile_space(intersected_indices, tile_index)
            value_indices = tuple(slice((s.start - n.start) // s.step, (s.stop - n.start - 1) // s.step + 1)
                                for s, n in zip(intersected_indices, indices))
            
            tile.values[tile_space_indices] += value[value_indices]
            if weights is not None:
                if tile.value_weights is None:
                    i = 0
                    value_weights_shape = []
                    for s, d in zip(weights.shape, self._shape):
                        if d is not None:
                            value_weights_shape.append(s)
                        else:
                            value_weights_shape.append(self._chunk_size[i])
                            i += 1
                    tile.value_weights = torch.zeros(value_weights_shape, dtype=torch.float32)
                tile.value_weights[tile_space_indices] += weights[value_indices]
            
            tile.modified = True

