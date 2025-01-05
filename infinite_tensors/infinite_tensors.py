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

from infinite_tensors.tensor_window import TensorWindow
from infinite_tensors.tilestore import MemoryTileStore, TileStore
from infinite_tensors.utils import normalize_slice, standardize_indices

# pixel space - Pixel space with all dimensions included
# window space - Each point is a window of pixels
# tile space - Each point is a tile of pixels

@dataclass
class InfinityTensorTile:
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
                 chunk_size: int|tuple[int, ...] = 512,
                 dtype: torch.dtype = torch.float32,
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
        self._shape = shape
        if isinstance(chunk_size, int):
            self._chunk_size = (chunk_size,) * sum(1 for dim in shape if dim is None)
        else:
            self._chunk_size = chunk_size
        self._dtype = dtype
        self._uuid = uuid.uuid4()
        self._operators = []
        self._store = tile_store or MemoryTileStore()
        
        self._dependency_windows = []
        
        self._f = f
        self._args = args or []
        for i in range(len(self._args)):
            if isinstance(self._args[i], InfiniteTensor):
                assert args_windows[i] is not None, f"Argument window must be provided for infinite tensors (arg {i})"
                self._args[i]._dependency_windows.append(args_windows[i])
            else:
                assert args_windows[i] is None, f"Argument window must not be provided for non-infinite tensors (arg {i})"
        self._kwargs = kwargs or {}
        for kwarg in self._kwargs:
            if isinstance(self._kwargs[kwarg], InfiniteTensor):
                assert kwargs_windows.get(kwarg) is not None, f"Argument window must be provided for infinite tensors (kwarg {kwarg})"
                self._kwargs[kwarg]._dependency_windows.append(kwargs_windows[kwarg])
            else:
                assert kwargs_windows.get(kwarg) is None, f"Argument window must not be provided for non-infinite tensors (kwarg {kwarg})"
        self._output_window = output_window
        self._args_windows = args_windows or []
        self._kwargs_windows = kwargs_windows or {}
        
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
        
        self._marked_for_cleanup = False
    
    @property
    def uuid(self) -> UUID:
        return self._uuid

    def _get_tile_key(self, tile_index: tuple[int, ...]) -> tuple[UUID, tuple[int, ...]]:
        return (self._uuid, tile_index)

    def _pixel_slices_to_tile_ranges(self, slices: tuple[slice, ...]) -> tuple[range, ...]:
        tile_ranges = []
        i = 0
        for j, pixel_range in enumerate(slices):
            if self._shape[j] is None:
                tile_ranges.append(range(pixel_range.start // self._chunk_size[i],
                                        (pixel_range.stop - 1) // self._chunk_size[i] + 1))
                i += 1
        return tuple(tile_ranges)
        
    def _intersect_slices(self, slices: tuple[slice, ...], tile_idx: tuple[int, ...]) -> tuple[slice, ...]:
        """Returns the intersection of the given pixel-space slices with the tile boundaries.

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
        
    def _translate_slices(self, slices: tuple[slice, ...], tile_idx: tuple[int, ...]) -> tuple[int, ...]:
        """Translate slices by (-tile index * chunk size)"""
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
        
        self._apply_f_range(indices)
        
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
            tile = self._store.get(self._get_tile_key(tile_index))
            intersected_indices = self._intersect_slices(indices, tile_index)
            tile_space_indices = self._translate_slices(intersected_indices, tile_index)
            output_indices = tuple(slice((s.start - n.start) // s.step, (s.stop - n.start + s.step - 1) // s.step)
                                   for s, n in zip(intersected_indices, indices))
            
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
        indices, collapse_dims = standardize_indices(self._shape, indices)
        tile_ranges = self._pixel_slices_to_tile_ranges(indices)
        
        self._apply_f_range(indices)
        
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
            tile = self._store.get(self._get_tile_key(tile_index))
            intersected_indices = self._intersect_slices(indices, tile_index)
            tile_space_indices = self._translate_slices(intersected_indices, tile_index)
            value_indices = tuple(slice((s.start - n.start) // s.step, (s.stop - n.start - 1) // s.step + 1)
                                for s, n in zip(intersected_indices, indices))
            
            tile.values[tile_space_indices] = value[value_indices]
            self._store.set(self._get_tile_key(tile_index), tile)
            
    def _add_op(self, indices: tuple[int|slice, ...], value: torch.Tensor, window_index: tuple[int, ...]):
        """Shortcut for self[indices] = self[indices] + value.
        Also tracks which windows have been processed.

        Args:
            indices (tuple[int | slice, ...]): Indices to add the value to.
            value (torch.Tensor): Value to add to the tensor.

        Raises:
            ValueError: If the value shape does not match the indexed shape.
        """
        indices, collapse_dims = standardize_indices(self._shape, indices)
        tile_ranges = self._pixel_slices_to_tile_ranges(indices)
        
        self._apply_f_range(indices)
        
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
            tile = self._store.get(self._get_tile_key(tile_index))
            if tile is None:
                tile = InfinityTensorTile(values=torch.zeros(self._tile_shape, dtype=self._dtype))
            intersected_indices = self._intersect_slices(indices, tile_index)
            tile_space_indices = self._translate_slices(intersected_indices, tile_index)
            value_indices = tuple(slice((s.start - n.start) // s.step, (s.stop - n.start - 1) // s.step + 1)
                                for s, n in zip(intersected_indices, indices))
            tile.values[tile_space_indices] += value[value_indices]
            self._store.set(self._get_tile_key(tile_index), tile)

    def _apply_f_range(self, pixel_range: tuple[slice, ...]):
        lowest_window_indexes = self._output_window.get_lowest_intersection(pixel_range)
        highest_window_indexes = self._output_window.get_highest_intersection(pixel_range)
        for idx in itertools.product(*[range(l, h+1) for l, h in zip(lowest_window_indexes, highest_window_indexes)]):
            self._apply_f(idx)

    def _apply_f(self, window_index: tuple[int, ...]) -> torch.Tensor:
        if self._store.is_window_processed(window_index):
            return
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
        self._add_op(self._output_window.get_bounds(window_index), output, window_index)
        
        for i, arg in enumerate(self._args):
            if isinstance(arg, InfiniteTensor):
                arg._mark_dependency_processed(self._args_windows[i], window_index)
        for kwarg in self._kwargs:
            if isinstance(self._kwargs[kwarg], InfiniteTensor):
                self._kwargs[kwarg]._mark_dependency_processed(self._kwargs_windows[kwarg], window_index)

    def _mark_dependency_processed(self, window, window_index: tuple[int, ...]):
        """Marks a window as processed for relevant tiles, incrementing their dependency_windows_processed counters.
        
        Args:
            window: The window that was processed
            window_index: The index of the window that was processed
        """
        tile_ranges = self._pixel_slices_to_tile_ranges(window.get_bounds(window_index))
        for tile_index in itertools.product(*tile_ranges):
            tile = self._store.get(self._get_tile_key(tile_index))
            tile.dependency_windows_processed += 1
            self._store.set(self._get_tile_key(tile_index), tile)
            if self._marked_for_cleanup and not self._is_tile_needed(tile_index):
                self._store.delete(self._get_tile_key(tile_index))
    
    def _is_tile_needed(self, tile_index: tuple[int, ...]) -> bool:
        """Returns True if the tile is depended on by any other InfiniteTensor."""
        tile_slices = self._get_tile_bounds(tile_index)
        
        total_windows = 0
        for dependency_window in self._dependency_windows:
            window_range = dependency_window.pixel_range_to_window_range(tile_slices)
            if window_range is None:
                continue
            num_windows = 1
            for r in window_range:
                num_windows *= len(range(r.start, r.stop))
            total_windows += num_windows
        
        return total_windows != self._store.get(self._get_tile_key(tile_index)).dependency_windows_processed

    def _get_tile_bounds(self, tile_index: tuple[int, ...]) -> tuple[slice, ...]:
        """Calculates the pixel space bounds of a tile given its index."""
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
        self.cleanup()
        return False  # Don't suppress any exceptions

    def __del__(self):
        self.cleanup()
                
    def cleanup(self):
        """Clean up unneeded resources used by this tensor."""
        if not self._marked_for_cleanup:
            for uuid, tile_index in self._store.keys():
                if not self._is_tile_needed(tile_index):
                    self._store.delete(self._get_tile_key(tile_index))
            self._marked_for_cleanup = True
