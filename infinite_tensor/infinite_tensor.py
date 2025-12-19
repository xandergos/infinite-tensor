from dataclasses import dataclass
import logging
from typing import Any, Callable, Iterable, Literal, Optional, Union
import warnings
import numpy as np
import torch
import itertools
import uuid

from infinite_tensor.tensor_window import TensorWindow
from infinite_tensor.tilestore import TileStore
from infinite_tensor.utils import standardize_indices

# COORDINATE SYSTEM DEFINITIONS:
# pixel space - Pixel space with all dimensions included (raw tensor coordinates)
# window space - Each point is a window of pixels (used for sliding window operations)
# tile space - Each point is a tile of pixels (internal tiling for memory management)

# CONSTANTS
DEFAULT_TILE_SIZE = 512
DEFAULT_DTYPE = torch.float32
DEFAULT_CACHE_METHOD = 'indirect'
DEFAULT_CACHE_LIMIT_BYTES = 10 * 1024 * 1024  # 10MB

# ERROR MESSAGES
TILE_DELETED_ERROR_MSG = "Tile has been deleted. This indicates either a bug or an attempt to access a tensor after cleanup."
SHAPE_MISMATCH_ERROR_MSG = "Value shape {actual} does not match indexed shape {expected}"
OUTPUT_SHAPE_ERROR_MSG = "Function output shape {actual} does not match expected window shape {expected}"
DEVICE_MISMATCH_ERROR_MSG = "Device mismatch: value is on {actual}, but infinite tensors require CPU tensors."

# Set up logging
logger = logging.getLogger(__name__)

# CUSTOM EXCEPTIONS
class InfiniteTensorError(Exception):
    """Base exception for infinite tensor operations."""
    pass

class TileAccessError(InfiniteTensorError):
    """Raised when trying to access a deleted or invalid tile."""
    pass

class ShapeMismatchError(InfiniteTensorError):
    """Raised when tensor shapes don't match expected dimensions."""
    pass

class ValidationError(InfiniteTensorError):
    """Raised when parameter validation fails."""
    pass


def _dtype_to_str(dtype: torch.dtype) -> str:
    """Serialize a torch.dtype to a compact string (e.g., 'float32')."""
    s = str(dtype)
    if s.startswith("torch."):
        return s.split(".")[-1]
    return s


def _str_to_dtype(name: str) -> torch.dtype:
    """Deserialize a dtype string back to torch.dtype."""
    dt = getattr(torch, name, None)
    if dt is None:
        raise ValueError(f"Unknown torch dtype: {name}")
    return dt


## Removed InfiniteTensorMeta dataclass.

def _validate_shape(shape: tuple) -> None:
    """Validate tensor shape specification."""
    if not isinstance(shape, tuple):
        raise ValidationError(f"Shape must be a tuple, got {type(shape)}")
    if len(shape) == 0:
        raise ValidationError("Shape cannot be empty")
    for i, dim in enumerate(shape):
        if dim is not None and (not isinstance(dim, int) or dim <= 0):
            raise ValidationError(f"Dimension {i} must be None or positive integer, got {dim}")

def _validate_tile_size(tile_size: Union[int, tuple], infinite_dims: int) -> None:
    """Validate tile size parameter."""
    if isinstance(tile_size, int):
        if tile_size <= 0:
            raise ValidationError(f"Tile size must be positive, got {tile_size}")
    elif isinstance(tile_size, tuple):
        if len(tile_size) != infinite_dims:
            raise ValidationError(f"Tile size tuple length {len(tile_size)} must match infinite dimensions {infinite_dims}")
        if any(c <= 0 for c in tile_size):
            raise ValidationError(f"All tile sizes must be positive, got {tile_size}")
    else:
        raise ValidationError(f"Tile size must be int or tuple, got {type(tile_size)}")

def _validate_function(f: Callable) -> None:
    """Validate the generating function."""
    if not callable(f):
        raise ValidationError(f"Function must be callable, got {type(f)}")

def _validate_window_args(args: tuple, args_windows) -> None:
    """Validate window argument consistency for positional args only."""
    if args_windows is not None and len(args_windows) != len(args):
        raise ValidationError(f"args_windows length {len(args_windows)} must match args length {len(args)}")

def _validate_cache_method(cache_method: str, cache_limit: Optional[int]) -> None:
    """Validate cache method and limit parameters."""
    if cache_method not in ('indirect', 'direct'):
        raise ValidationError(f"cache_method must be 'indirect' or 'direct', got '{cache_method}'")
    if cache_method == 'direct' and cache_limit is not None:
        if not isinstance(cache_limit, int) or cache_limit <= 0:
            raise ValidationError(f"cache_limit must be a positive integer or None, got {cache_limit}")

def _validate_tensor_windows(
    tensor_shape: tuple[int|None, ...],
    output_window: TensorWindow,
    arg_tensors: list,
    arg_windows: list,
) -> None:
    """Validate that window dimensionality matches corresponding tensor dimensionality.

    Ensures:
    - output_window dims == len(tensor_shape)
    - For each arg tensor/window pair: window dims == len(arg.shape)
    """
    # Validate output window dimensionality
    if len(output_window.size) != len(tensor_shape):
        raise ValidationError(
            f"output_window has {len(output_window.size)} dims but tensor has {len(tensor_shape)} dims"
        )

    # Validate argument windows dimensionality
    for i, (arg_tensor, arg_window) in enumerate(zip(arg_tensors, arg_windows)):
        if arg_window is None:
            # Higher-level code requires arg windows for all args; keep message focused on dims
            continue
        arg_tensor_shape = arg_tensor.shape
        if arg_tensor_shape is None:
            continue
        if len(arg_window.size) != len(arg_tensor_shape):
            raise ValidationError(
                f"args_windows[{i}] has {len(arg_window.size)} dims but corresponding arg tensor has {len(arg_tensor_shape)} dims"
            )

@dataclass
class InfinityTensorTile:
    """A single tile storing tensor data."""

    values: torch.Tensor

class InfiniteTensor:
    @classmethod
    def from_existing(cls, store: TileStore, tensor_id: str, f: Optional[Callable] = None) -> "InfiniteTensor":
        """Return the existing tensor instance registered in the store.
        
        Args:
            store: TileStore containing the tensor
            tensor_id: Unique identifier for the tensor
            f: Optional computation function. Required if tensor is not in cache
               and needs to be reconstructed from metadata.
               
        Returns:
            InfiniteTensor instance
        """
        return store.get_tensor(tensor_id, f=f)
    
    def __init__(self,
                 shape: tuple[int|None, ...],
                 f: Callable,
                 output_window: TensorWindow,
                 args: tuple = None,
                 args_windows = None,
                 tile_size: Union[int, tuple[int, ...]] = DEFAULT_TILE_SIZE,
                 dtype: torch.dtype = DEFAULT_DTYPE,
                 tile_store: Optional[TileStore] = None,
                 tensor_id: Optional[Any] = None,
                 batch_size: Optional[int] = None,
                 cache_method: Literal['indirect', 'direct'] = DEFAULT_CACHE_METHOD,
                 cache_limit: Optional[int] = DEFAULT_CACHE_LIMIT_BYTES,
                 _created_via_store: bool = False):
        """Initialize an InfiniteTensor.

        An InfiniteTensor represents a theoretically infinite tensor that is processed in tiles.
        Operations can be performed on the tensor in a sliding window manner without loading
        the entire tensor into memory.

        Args:
            tensor_id: Any unique identifier for the tensor. This is used to identify the tensor in the TileStore.
                       If None, a random UUID string will be generated.
            tile_store: TileStore to use for storing tiles.
            shape: Shape of the tensor. Use None to indicate that the tensor is infinite in that dimension. For example (3, 10, None, None) 
                   indicates a 4-dimensional tensor with the first two dimensions of size 3 and 10, and the last two dimensions being infinite.
                   This might be used to represent a batch of 3 images, with 10 channels and an infinite width and height.
            f: Callable[[Any, *Any], torch.Tensor]: A function that takes a context and optional positional arguments and returns a tensor. 
               The function signature should be f(ctx, *args), where ctx is the index of the window that is being processed.
            args: Positional arguments to pass to the function f. All must be InfiniteTensor if provided.
            args_windows: Optional positional argument windows specific to window processing.
            tile_size: Size of each tile. Can be an integer for uniform tile size, or tuple of integers to specify a different tile size for each dimension.
                       The number of 'None' values in 'shape' must match the number of dimensions in 'tile_size' if it is a tuple.
                       Ignored when cache_method='direct'.
            dtype: PyTorch data type for the tensor (default: torch.float32)
            batch_size: Number of tensors to batch together. If None, no batching is done.
            cache_method: Caching strategy - 'indirect' (default) stores tiles, 'direct' caches window outputs.
            cache_limit: Maximum cache size in bytes for direct caching (default: 10MB). None for unlimited.
        """
        # Enforce creation via store.get_or_create
        if not _created_via_store:
            raise ValidationError("InfiniteTensor must be created via TileStore.get_or_create(..)")

        # Validate parameters (will be skipped if reconstructing existing tensor)
        _validate_shape(shape)
        _validate_function(f)
        _validate_cache_method(cache_method, cache_limit)
        
        infinite_dims = sum(1 for dim in shape if dim is None)
        if cache_method == 'indirect':
            _validate_tile_size(tile_size, infinite_dims)
        _validate_window_args(args or (), args_windows)
        
        # Setup store and ID
        if tile_store is None:
            raise ValidationError("A TileStore instance is required")
        self._store = tile_store
        # Ensure tensor id is a string
        self._uuid = str(tensor_id) if tensor_id is not None else str(uuid.uuid4())

        # Normalize windows and args
        normalized_args = list(args or [])
        normalized_args_windows = list(args_windows) if args_windows is not None else [None] * len(normalized_args)

        # Validate window dimensionality against tensors
        _validate_tensor_windows(shape, output_window, normalized_args, normalized_args_windows)

        # Compute tile size tuple across infinite dims
        if isinstance(tile_size, int):
            tile_size_tuple = (tile_size,) * infinite_dims
        else:
            tile_size_tuple = tile_size

        # Compute tile shape
        tile_shape_list = []
        i = 0
        for dim in shape:
            if dim is None:
                tile_shape_list.append(tile_size_tuple[i])
                i += 1
            else:
                tile_shape_list.append(dim)
        tile_shape = tuple(tile_shape_list)
        _elem_size = torch.empty((), dtype=dtype).element_size()
        tile_bytes = int(np.prod(list(tile_shape))) * _elem_size

        # Enforce that all args are InfiniteTensor in the same store, and store only their UUIDs
        arg_ids = []
        for i, arg in enumerate(normalized_args):
            if not isinstance(arg, InfiniteTensor):
                raise ValidationError("All positional args must be InfiniteTensor instances")
            if arg._store is not self._store:
                raise ValidationError("All related tensors must use the same TileStore instance")
            assert normalized_args_windows[i] is not None, f"Argument window must be provided for infinite tensors (arg {i})"
            arg_ids.append(arg.uuid)

        # Inline metadata on this instance
        self._shape = shape
        self._tile_size = tile_size_tuple
        self._dtype = dtype
        self._f = f
        self._args = arg_ids
        self._args_windows = normalized_args_windows
        self._output_window = output_window
        self._tile_shape = tile_shape
        self._tile_bytes = tile_bytes
        self._batch_size = batch_size
        self._cache_method = cache_method
        self._cache_limit = cache_limit
        
        # Register this tensor instance in the store
        self._store.register_tensor_meta(self._uuid, self.to_json())
    
    @property
    def shape(self) -> tuple[int|None, ...]:
        return self._shape
    
    @property
    def uuid(self) -> str:
        return self._uuid

    def _meta(self) -> "InfiniteTensor":
        # For backward compatibility in internal methods that expect a meta-like object,
        # return self, exposing required fields via properties below.
        return self

    # Expose meta fields as properties for internal access via self._meta()
    @property
    def tile_size(self) -> tuple[int, ...]:
        return self._tile_size

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def f(self) -> Callable:
        return self._f

    @property
    def args(self) -> list:
        return self._args

    @property
    def args_windows(self) -> list:
        return self._args_windows

    @property
    def output_window(self) -> TensorWindow:
        return self._output_window

    @property
    def tile_shape(self) -> tuple[int, ...]:
        return self._tile_shape

    @property
    def tile_bytes(self) -> int:
        return self._tile_bytes
    
    @property
    def batch_size(self) -> int:
        return self._batch_size
    
    @property
    def cache_method(self) -> str:
        return self._cache_method
    
    @property
    def cache_limit(self) -> Optional[int]:
        return self._cache_limit

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
        for idx, shape_dim in zip(indices, self.shape):
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
            
        value = torch.as_tensor(value, dtype=self.dtype)
        if value.shape != expected_output_shape:
            raise ShapeMismatchError(SHAPE_MISMATCH_ERROR_MSG.format(actual=value.shape, expected=expected_output_shape))
            
        # Expand value to match indexed shape if needed
        if collapse_dims:
            value = value.reshape(expected_shape)
            
        return value

    def _pixel_slices_to_tile_ranges(self, slices: tuple[slice, ...]) -> tuple[slice, ...]:
        """Convert pixel-space slices to tile-space ranges.
        
        Takes pixel coordinates and determines which tiles contain those pixels.
        Only processes infinite dimensions (where shape[i] is None).
        
        Args:
            slices: Tuple of slices in pixel space
            
        Returns:
            Tuple of slices indicating which tiles are needed for each infinite dimension
            
        Example:
            If tile_size is 512 and we want pixels [100:1500], this returns
            slice(0, 3) since we need tiles 0, 1, and 2 to cover that pixel range.
        """
        tile_ranges = []
        i = 0
        for j, pixel_range in enumerate(slices):
            if self.shape[j] is None:
                start = None if pixel_range.start is None else pixel_range.start // self.tile_size[i]
                stop = None if pixel_range.stop is None else (pixel_range.stop - 1) // self.tile_size[i] + 1
                
                tile_ranges.append(slice(start, stop))
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
            if self.shape[i] is None:
                # Calculate tile boundaries
                tile_start = tile_idx[infinite_dim] * self.tile_size[infinite_dim]
                tile_end = (tile_idx[infinite_dim] + 1) * self.tile_size[infinite_dim]
                
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
            If tile_idx=(1,) and tile_size=(512,), then pixel slice [600:700]
            becomes tile-local slice [88:188] (600-512:700-512)
        """
        infinite_dim = 0
        output = []
        for i, s in enumerate(slices):
            if self.shape[i] is None:
                output.append(slice(s.start - tile_idx[infinite_dim] * self.tile_size[infinite_dim],
                                    s.stop - tile_idx[infinite_dim] * self.tile_size[infinite_dim],
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
        indices, collapse_dims = standardize_indices(self.shape, indices)
        
        logger.debug(f"Accessing tensor slice with indices: {indices}")
        self._apply_f_range([indices])
        
        # Calculate output shape
        output_shape = self._calculate_indexed_shape(indices)
        logger.debug(f"Calculated output shape: {output_shape}")
        
        if self._cache_method == 'direct':
            result = self._getitem_direct(indices, output_shape, collapse_dims)
            self._store.evict_cache_for(self._uuid, self._cache_limit)
            return result
        else:
            return self._getitem_indirect(indices, output_shape, collapse_dims)
    
    def _getitem_indirect(self, indices: tuple[slice, ...], output_shape: list[int], 
                          collapse_dims: list[int]) -> torch.Tensor:
        """Get values using indirect caching (tile-based storage)."""
        tile_ranges = [range(s.start, s.stop) for s in self._pixel_slices_to_tile_ranges(indices)]
        
        output_tensor = torch.empty(output_shape, dtype=self.dtype)
        for tile_index in itertools.product(*tile_ranges):
            tile = self._store.get_tile_for(self._uuid, tile_index)
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
    
    def _getitem_direct(self, indices: tuple[slice, ...], output_shape: list[int],
                        collapse_dims: list[int]) -> torch.Tensor:
        """Get values using direct caching (window output-based storage)."""
        # Find which windows intersect with the requested region
        lowest = self.output_window.get_lowest_intersection(indices)
        highest = self.output_window.get_highest_intersection(indices)
        window_ranges = [range(l, h + 1) for l, h in zip(lowest, highest)]
        
        # Use zeros and += to correctly accumulate overlapping window contributions
        output_tensor = torch.zeros(output_shape, dtype=self.dtype)
        for window_index in itertools.product(*window_ranges):
            cached_output = self._store.get_cached_window_for(self._uuid, window_index)
            if cached_output is None:
                raise TileAccessError(f"Window {window_index} not found in cache")
            
            # Get the pixel bounds of this window
            window_bounds = self.output_window.get_bounds(window_index)
            
            # Calculate intersection of requested indices with window bounds
            intersected = []
            for idx_slice, win_slice in zip(indices, window_bounds):
                start = max(idx_slice.start, win_slice.start)
                stop = min(idx_slice.stop, win_slice.stop)
                intersected.append(slice(start, stop, idx_slice.step))
            
            # Skip if no intersection
            if any(s.start >= s.stop for s in intersected):
                continue
            
            # Calculate indices within the cached window output
            window_local_indices = tuple(
                slice((s.start - w.start), (s.stop - w.start), s.step)
                for s, w in zip(intersected, window_bounds)
            )
            
            # Calculate indices within the output tensor
            output_indices = tuple(
                slice((s.start - n.start) // (s.step or 1), 
                      (s.stop - n.start + (s.step or 1) - 1) // (s.step or 1))
                for s, n in zip(intersected, indices)
            )
            
            output_tensor[output_indices] += cached_output[window_local_indices]
            
        if collapse_dims:
            target_shape = tuple(s for i, s in enumerate(output_shape) if i not in collapse_dims)
            return output_tensor.reshape(target_shape)
        return output_tensor
            
    def __setitem__(self, indices: tuple[int|slice, ...], value: torch.Tensor):
        """Set a slice of the tensor.
        
        .. deprecated::
            __setitem__ is deprecated and will be removed in a future version.
        
        Args:
            indices: Tuple of indices/slices to set. Can include integers, slices, and ellipsis.
                    For infinite dimensions, slices can extend beyond defined regions.
            value: Value to set at the specified indices. Must match the shape of the indexed region.
                  Will be automatically converted to a tensor with the correct dtype.
                  
        Raises:
            ValueError: If the value shape does not match the indexed shape.
            InfiniteTensorError: If using direct caching (not supported).
        """
        warnings.warn(
            "__setitem__ is deprecated and will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2
        )
        if self._cache_method == 'direct':
            raise InfiniteTensorError("__setitem__ is not supported with cache_method='direct'")
        
        indices, collapse_dims = standardize_indices(self.shape, indices)
        tile_ranges = [range(s.start, s.stop) for s in self._pixel_slices_to_tile_ranges(indices)]
        
        self._apply_f_range([indices])
        
        # Validate and prepare value
        value = self._validate_and_prepare_value(indices, value, collapse_dims)
            
        # Set values in tiles
        for tile_index in itertools.product(*tile_ranges):
            tile = self._store.get_tile_for(self._uuid, tile_index)
            if tile is None:
                raise TileAccessError(TILE_DELETED_ERROR_MSG)
            intersected_indices = self._intersect_slices(indices, tile_index)
            tile_space_indices = self._translate_slices(intersected_indices, tile_index)
            value_indices = tuple(slice((s.start - n.start) // s.step, (s.stop - n.start - 1) // s.step + 1)
                                for s, n in zip(intersected_indices, indices))
            
            tile.values[tile_space_indices] = value[value_indices]
            self._store.set_tile_for(self._uuid, tile_index, tile)
            
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
        indices, collapse_dims = standardize_indices(self.shape, indices)
        tile_ranges = [range(s.start, s.stop) for s in self._pixel_slices_to_tile_ranges(indices)]
        
        # Validate and prepare value
        value = self._validate_and_prepare_value(indices, value, collapse_dims)
            
        # Set values in tiles
        for tile_index in itertools.product(*tile_ranges):
            tile = self._store.get_tile_for(self._uuid, tile_index)
            if tile is None:
                tile = InfinityTensorTile(values=torch.zeros(self.tile_shape, dtype=self.dtype))

            intersected_indices = self._intersect_slices(indices, tile_index)
            tile_space_indices = self._translate_slices(intersected_indices, tile_index)
            value_indices = tuple(slice((s.start - n.start) // s.step, (s.stop - n.start - 1) // s.step + 1)
                                for s, n in zip(intersected_indices, indices))
            
            # Assert tile values and value are on same device
            if tile.values.device != value.device:
                raise ValueError(DEVICE_MISMATCH_ERROR_MSG.format(actual=value.device))
            tile.values[tile_space_indices] += value[value_indices]
            self._store.set_tile_for(self._uuid, tile_index, tile)

    def _apply_f_range(self, pixel_range: Iterable[tuple[slice, ...]]):
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
        # Process windows in batches
        all_indices = itertools.chain.from_iterable(
            itertools.product(*[range(l, h+1) for l, h in zip(
                self.output_window.get_lowest_intersection(sub_pixel_range),
                self.output_window.get_highest_intersection(sub_pixel_range)
            )])
            for sub_pixel_range in pixel_range
        )
        self._apply_f(all_indices)

    def _apply_f(self, window_indices: Iterable[tuple[int, ...]]):
        """Apply the generating function to a specific window.
        
        This method orchestrates the application of the user-provided function to
        generate data for a specific window. It handles dependency resolution,
        argument preparation, and result accumulation.
        
        Args:
            window_index: N-dimensional index of the window in window space
            
        Returns:
            None (results are stored in tiles via _add_op or cached directly)
            
        Internal Process:
            1. Check if window is already processed (skip if so)
            2. Prepare arguments by slicing dependent tensors using their windows
            3. Call user function with window context and prepared arguments
            4. Validate output shape matches expected window shape
            5. For indirect caching: accumulate result into tiles using _add_op
               For direct caching: cache the output directly
            6. Mark dependencies as processed for cleanup tracking
        """
        valid_window_indices = set()
        for window_index in window_indices:
            # For direct caching, check the window cache; for indirect, check the store
            if self._cache_method == 'direct':
                if self._store.is_window_cached_for(self._uuid, window_index):
                    logger.debug(f"Window {window_index} already cached, skipping")
                    continue
            else:
                if self._store.is_window_processed_for(self._uuid, window_index):
                    logger.debug(f"Window {window_index} already processed, skipping")
                    continue
            
            valid_window_indices.add(window_index)
        valid_window_indices = list(valid_window_indices)
        
        # Track which upstream windows were used (for cache prioritization)
        used_windows_by_tensor: dict[str, list[tuple[int, ...]]] = {}
        
        def track_upstream_windows():
            """Calculate which upstream windows will be accessed."""
            for i, arg_window in enumerate(self.args_windows):
                upstream = self._store.get_tensor(self.args[i])
                if upstream._cache_method != 'direct':
                    continue
                for window_index in valid_window_indices:
                    bounds = arg_window.get_bounds(window_index)
                    lowest = upstream.output_window.get_lowest_intersection(bounds)
                    highest = upstream.output_window.get_highest_intersection(bounds)
                    for upstream_window in itertools.product(
                        *[range(l, h + 1) for l, h in zip(lowest, highest)]
                    ):
                        used_windows_by_tensor.setdefault(upstream._uuid, []).append(upstream_window)
        
        if self._cache_method == 'direct':
            track_upstream_windows()
            
        # Pre-process arguments
        for i, arg_window in enumerate(self.args_windows):
            upstream = self._store.get_tensor(self.args[i])
            arg_window = self.args_windows[i]
            pixel_ranges = [arg_window.get_bounds(window_index) for window_index in valid_window_indices]
            upstream._apply_f_range(pixel_ranges)
        
        if self.batch_size is None:
            for window_index in valid_window_indices:
                # Resolve arg IDs to tensors lazily
                args = []
                for i, arg_window in enumerate(self.args_windows):
                    upstream = self._store.get_tensor(self.args[i])
                    arg_window = self.args_windows[i]
                    args.append(upstream[arg_window.get_bounds(window_index)])
                with torch.no_grad():
                    output = self.f(window_index, *args)
                
                # Verify output shape matches the expected window shape
                expected_shape = self.output_window.size
                if tuple(output.shape) != tuple(expected_shape):
                    raise ShapeMismatchError(OUTPUT_SHAPE_ERROR_MSG.format(actual=output.shape, expected=expected_shape))
                
                if self._cache_method == 'direct':
                    self._store.cache_window_for(self._uuid, window_index, output)
                else:
                    self._add_op(self.output_window.get_bounds(window_index), output)
                    self._store.mark_window_processed_for(self._uuid, window_index)
            
                logger.debug(f"Processed window {window_index}")
        else:
            def apply_batch(batch_window_indices: list[tuple[int, ...]]):
                args = [[] for _ in range(len(self.args))]
                # Resolve args
                for window_index in batch_window_indices:
                    for i, arg_window in enumerate(self.args_windows):
                        upstream = self._store.get_tensor(self.args[i])
                        arg_window = self.args_windows[i]
                        args[i].append(upstream[arg_window.get_bounds(window_index)])
                
                with torch.no_grad():
                    outputs = self.f(batch_window_indices, *args)
                    
                for window_index, output in zip(batch_window_indices, outputs):
                    # Verify output shape matches the expected window shape
                    expected_shape = self.output_window.size
                    if tuple(output.shape) != tuple(expected_shape):
                        raise ShapeMismatchError(OUTPUT_SHAPE_ERROR_MSG.format(actual=output.shape, expected=expected_shape))
                    
                    if self._cache_method == 'direct':
                        self._store.cache_window_for(self._uuid, window_index, output)
                    else:
                        self._add_op(self.output_window.get_bounds(window_index), output)
                        self._store.mark_window_processed_for(self._uuid, window_index)
                
                    logger.debug(f"Processed window {window_index}")
                
            batched_window_indices = []
            for window_index in valid_window_indices:
                batched_window_indices.append(window_index)
                if len(batched_window_indices) == self.batch_size:
                    apply_batch(batched_window_indices)
                    batched_window_indices = []
            if batched_window_indices:
                apply_batch(batched_window_indices)
        
        # Reorder caches: first promote USED windows, then GENERATED windows
        # This ensures generated windows have highest priority (evicted last)
        if self._cache_method == 'direct':
            for tensor_id, windows in used_windows_by_tensor.items():
                self._store.promote_windows_for(tensor_id, windows)
            self._store.promote_windows_for(self._uuid, valid_window_indices)
            

    def _get_tile_bounds(self, tile_index: tuple[int, ...]) -> tuple[slice, ...]:
        """Calculate the pixel-space boundaries of a tile.
        
        Converts a tile index in tile space to the corresponding pixel coordinates
        that the tile covers. Essential for dependency tracking and tile management.
        
        Args:
            tile_index: N-dimensional index of the tile in tile space
            
        Returns:
            Tuple of slices defining the pixel boundaries covered by this tile
            
        Example:
            For tile_index=(1, 2) with tile_size=(512, 512), returns
            (slice(512, 1024), slice(1024, 1536)) for the pixel bounds
        """
        tile_slices = []
        infinite_dim = 0
        for i, dim in enumerate(self.shape):
            if dim is None:
                start = tile_index[infinite_dim] * self.tile_size[infinite_dim]
                stop = (tile_index[infinite_dim] + 1) * self.tile_size[infinite_dim]
                tile_slices.append(slice(start, stop))
                infinite_dim += 1
            else:
                tile_slices.append(slice(0, dim))
        return tuple(tile_slices)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False  # Don't suppress any exceptions

    def to_json(self) -> dict:
        """Serialize this tensor's metadata to JSON."""
        def window_to_dict(w: Optional[TensorWindow]):
            return None if w is None else w.to_dict()

        return {
            'shape': list(self.shape),
            'tile_size': list(self.tile_size),
            'dtype': _dtype_to_str(self.dtype),
            'args': [str(a) for a in self.args],
            'args_windows': [window_to_dict(w) for w in self.args_windows],
            'output_window': self.output_window.to_dict(),
            'tile_shape': list(self.tile_shape),
            'tile_bytes': int(self.tile_bytes),
            'cache_method': self.cache_method,
            'cache_limit': self.cache_limit,
        }
        
