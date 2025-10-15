from dataclasses import dataclass
import logging
from typing import Any, Callable, Optional, Union
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

def _validate_window_args(args: tuple, args_windows) -> None:
    """Validate window argument consistency for positional args only."""
    if args_windows is not None and len(args_windows) != len(args):
        raise ValidationError(f"args_windows length {len(args_windows)} must match args length {len(args)}")

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
    num_windows: int
    num_windows_processed: int = 0

class InfiniteTensor:
    @classmethod
    def from_existing(cls, store: TileStore, tensor_id: str) -> "InfiniteTensor":
        """Return the existing tensor instance registered in the store."""
        return store.get_tensor_meta(tensor_id)
    def __init__(self,
                 shape: tuple[int|None, ...],
                 f: Callable,
                 output_window: TensorWindow,
                 args: tuple = None,
                 args_windows = None,
                 chunk_size: Union[int, tuple[int, ...]] = DEFAULT_CHUNK_SIZE,
                 dtype: torch.dtype = DEFAULT_DTYPE,
                 tile_store: Optional[TileStore] = None,
                 tensor_id: Optional[Any] = None,
                 _created_via_store: bool = False):
        """Initialize an InfiniteTensor.

        An InfiniteTensor represents a theoretically infinite tensor that is processed in chunks.
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
            chunk_size: Size of each chunk. Can be an integer for uniform chunk size, or tuple of integers to specify a different chunk size for each dimension.
                        The number of 'None' values in 'shape' must match the number of dimensions in 'chunk_size' if it is a tuple.
            dtype: PyTorch data type for the tensor (default: torch.float32)
        """
        # Enforce creation via store.get_or_create
        if not _created_via_store:
            raise ValidationError("InfiniteTensor must be created via TileStore.get_or_create(..)")

        # Validate parameters (will be skipped if reconstructing existing tensor)
        _validate_shape(shape)
        _validate_function(f)
        
        infinite_dims = sum(1 for dim in shape if dim is None)
        _validate_chunk_size(chunk_size, infinite_dims)
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

        # Compute chunk size tuple across infinite dims
        if isinstance(chunk_size, int):
            chunk_tuple = (chunk_size,) * infinite_dims
        else:
            chunk_tuple = chunk_size

        # Compute tile shape
        tile_shape_list = []
        i = 0
        for dim in shape:
            if dim is None:
                tile_shape_list.append(chunk_tuple[i])
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
        self._chunk_size = chunk_tuple
        self._dtype = dtype
        self._f = f
        self._args = arg_ids
        self._args_windows = normalized_args_windows
        self._output_window = output_window
        self._tile_shape = tile_shape
        self._tile_bytes = tile_bytes
        self._dependency_windows = []
        self._marked_for_cleanup = False
        self._fully_cleaned = False

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
    def chunk_size(self) -> tuple[int, ...]:
        return self._chunk_size

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
    def dependency_windows(self) -> list:
        return self._dependency_windows
    
    @dependency_windows.setter
    def dependency_windows(self, value: list):
        self._dependency_windows = value

    @property
    def marked_for_cleanup(self) -> bool:
        return self._marked_for_cleanup
    @marked_for_cleanup.setter
    def marked_for_cleanup(self, value: bool):
        self._marked_for_cleanup = bool(value)

    @property
    def fully_cleaned(self) -> bool:
        return self._fully_cleaned
    @fully_cleaned.setter
    def fully_cleaned(self, value: bool):
        self._fully_cleaned = bool(value)

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

    def _get_tile_key(self, tile_index: tuple[int, ...]) -> tuple[str, tuple[int, ...]]:
        """Generate a unique key for storing/retrieving a tile.
        
        Combines the tensor's UUID with the tile index to create a globally unique
        identifier for this specific tile.
        
        Args:
            tile_index: N-dimensional index of the tile in tile space
            
        Returns:
            Tuple containing (tensor_uuid, tile_index) for unique identification
        """
        return (self._uuid, tile_index)

    def _pixel_slices_to_tile_ranges(self, slices: tuple[slice, ...]) -> tuple[slice, ...]:
        """Convert pixel-space slices to tile-space ranges.
        
        Takes pixel coordinates and determines which tiles contain those pixels.
        Only processes infinite dimensions (where shape[i] is None).
        
        Args:
            slices: Tuple of slices in pixel space
            
        Returns:
            Tuple of slices indicating which tiles are needed for each infinite dimension
            
        Example:
            If chunk_size is 512 and we want pixels [100:1500], this returns
            slice(0, 3) since we need tiles 0, 1, and 2 to cover that pixel range.
        """
        tile_ranges = []
        i = 0
        for j, pixel_range in enumerate(slices):
            if self.shape[j] is None:
                start = None if pixel_range.start is None else pixel_range.start // self.chunk_size[i]
                stop = None if pixel_range.stop is None else (pixel_range.stop - 1) // self.chunk_size[i] + 1
                
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
                tile_start = tile_idx[infinite_dim] * self.chunk_size[infinite_dim]
                tile_end = (tile_idx[infinite_dim] + 1) * self.chunk_size[infinite_dim]
                
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
            if self.shape[i] is None:
                output.append(slice(s.start - tile_idx[infinite_dim] * self.chunk_size[infinite_dim],
                                    s.stop - tile_idx[infinite_dim] * self.chunk_size[infinite_dim],
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
        tile_ranges = [range(s.start, s.stop) for s in self._pixel_slices_to_tile_ranges(indices)]
        
        logger.debug(f"Accessing tensor slice with indices: {indices}")
        self._apply_f_range(indices)
        
        # Calculate output shape
        output_shape = self._calculate_indexed_shape(indices)
        logger.debug(f"Calculated output shape: {output_shape}")
                
        # Create output tensor
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
        if self.dependency_windows or self.marked_for_cleanup:
            raise DependencyError(DEPENDENCY_ERROR_MSG)
        
        indices, collapse_dims = standardize_indices(self.shape, indices)
        tile_ranges = [range(s.start, s.stop) for s in self._pixel_slices_to_tile_ranges(indices)]
        
        self._apply_f_range(indices)
        
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

        Additionally, increments the dependency_windows_processed counter for the tiles.

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
        
        #self._apply_f_range(indices)
        
        # Validate and prepare value
        value = self._validate_and_prepare_value(indices, value, collapse_dims)
            
        # Set values in tiles
        for tile_index in itertools.product(*tile_ranges):
            tile = self._store.get_tile_for(self._uuid, tile_index)
            if tile is None:
                tile = InfinityTensorTile(values=torch.zeros(self.tile_shape, dtype=self.dtype), num_windows=self.count_windows(tile_index))
            tile.num_windows_processed += 1
            if tile.num_windows_processed == tile.num_windows:
                for i, arg in enumerate(self.args):
                    upstream = self._store.get_tensor(arg)
                    upstream._alert_dependent_used(list(self._get_dependency_tiles(tile_index)))

            intersected_indices = self._intersect_slices(indices, tile_index)
            tile_space_indices = self._translate_slices(intersected_indices, tile_index)
            value_indices = tuple(slice((s.start - n.start) // s.step, (s.stop - n.start - 1) // s.step + 1)
                                for s, n in zip(intersected_indices, indices))
            
            # Assert tile values and value are on same device
            if tile.values.device != value.device:
                raise ValueError(DEVICE_MISMATCH_ERROR_MSG.format(actual=value.device))
            tile.values[tile_space_indices] += value[value_indices]
            self._store.set_tile_for(self._uuid, tile_index, tile)

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
        lowest_window_indexes = self.output_window.get_lowest_intersection(pixel_range)
        highest_window_indexes = self.output_window.get_highest_intersection(pixel_range)
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
        if self._store.is_window_processed_for(self._uuid, window_index):
            logger.debug(f"Window {window_index} already processed, skipping")
            return
        
        logger.debug(f"Processing window {window_index}")
        self._store.mark_window_processed_for(self._uuid, window_index)
        
        # Resolve arg IDs to tensors lazily
        args = []
        for i, arg_window in enumerate(self.args_windows):
            upstream = self._store.get_tensor(self.args[i])
            arg_window = self.args_windows[i]
            args.append(upstream[arg_window.get_bounds(window_index)])
        output = self.f(window_index, *args)
        
        # Verify output shape matches the expected window shape
        expected_shape = self.output_window.size
        if tuple(output.shape) != tuple(expected_shape):
            raise ShapeMismatchError(OUTPUT_SHAPE_ERROR_MSG.format(actual=output.shape, expected=expected_shape))
        
        self._add_op(self.output_window.get_bounds(window_index), output)

    def count_windows(self, tile_index: tuple[int, ...]) -> int:
        """Count the number of windows that intersect a given tile."""
        tile_pixel_slices = self._get_tile_bounds(tile_index)
        window_slices = self.output_window.pixel_range_to_window_range(tile_pixel_slices)
        count = 1
        for s in window_slices:
            count *= s.stop - s.start
        return count


    def _alert_dependent_used(self, dependency_tiles: list[tuple[str, tuple[int, ...]]]):
        """Mark a dependent tile as used and potentially trigger cleanup.
        
        This is called when a dependent tensor processes a window that references
        this tensor. It increments the processing counter for affected tiles and
        triggers cleanup if tiles are no longer needed.
        
        Args:
            dependency_tiles: List of tuples containing the tensor ID and tile index of the dependency tiles
            
        Internal Process:
            1. Determine which tiles are affected by the processed window
            2. Increment dependency_windows_processed counter for each tile
            3. If tensor is marked for cleanup, delete tiles that are no longer needed
            4. Attempt full cleanup if all dependencies are satisfied
        """
        for tensor_id, dep_tile_index in dependency_tiles:
            if not self._is_tile_needed(dep_tile_index):
                self._store.delete_tile_for(tensor_id, dep_tile_index)
        self._full_cleanup()
    
    def _get_dependency_tiles(self, tile_index: tuple[int, ...]):
        """Get all tiles that this tile depends on."""
        tile_pixel_slices = self._get_tile_bounds(tile_index)
        output_window_slices = self.output_window.pixel_range_to_window_range(tile_pixel_slices)
        
        for arg, arg_window in zip(self.args, self.args_windows):
            tensor = self._store.get_tensor(arg)
            arg_pixel_slice = arg_window.get_bounds(output_window_slices)
            arg_tile_ranges = [range(s.start, s.stop) for s in tensor._pixel_slices_to_tile_ranges(arg_pixel_slice)]
            for arg_tile_index in itertools.product(*arg_tile_ranges):
                yield (tensor._uuid, arg_tile_index)

    def _get_dependent_tiles(self, tile_index: tuple[int, ...]):
        """Get all tiles that are dependent on a given tile.
        
        This is used to determine if all dependents of a tile are processed.
        """
        tile_pixel_slices = self._get_tile_bounds(tile_index)
        for tensor_id in self._store.get_dependents(self._uuid):
            tensor = self._store.get_tensor(tensor_id)
            for arg, arg_window in filter(lambda x: x[0] == self._uuid, zip(tensor.args, tensor.args_windows)):
                input_window_slices = arg_window.pixel_range_to_window_range(tile_pixel_slices)
                output_window_slices = arg_window.inverse_map_window_slices(input_window_slices, len(tensor.shape))
                dependent_pixel_slice = tensor.output_window.get_bounds(output_window_slices)
                dependent_tile_ranges = [range(s.start, s.stop) for s in tensor._pixel_slices_to_tile_ranges(dependent_pixel_slice)]
                for dep_tile_index in itertools.product(*dependent_tile_ranges):
                    yield tensor_id, dep_tile_index
    
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
            2. For each dependent tensor:
                - Get all arg_windows that reference this tensor
                - Find window slice that intersects with the tile
                - Find all tiles that intersect with that window slice
                - Check if any of those tiles are not processed
            3. Return False if all dependents are processed
        """
        for tensor_id, dep_tile_index in self._get_dependent_tiles(tile_index):
            tile = self._store.get_tile_for(tensor_id, dep_tile_index)
            if tile is None:
                return True
            if tile.num_windows_processed < tile.num_windows:
                return True
        return False

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
        for i, dim in enumerate(self.shape):
            if dim is None:
                start = tile_index[infinite_dim] * self.chunk_size[infinite_dim]
                stop = (tile_index[infinite_dim] + 1) * self.chunk_size[infinite_dim]
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
        try:
            self.mark_for_cleanup()
        except Exception:
            pass
                
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
        if not self.marked_for_cleanup:
            # Delete tiles that are no longer needed
            for tile_index in list(self._store.iter_tile_keys_for(self._uuid)):
                if not self._is_tile_needed(tile_index):
                    self._store.delete_tile_for(self._uuid, tile_index)
            self._full_cleanup()
            self.marked_for_cleanup = True
            
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
        if self.marked_for_cleanup and (next(self._store.iter_tile_keys_for(self._uuid), None) is None) and len(self.dependency_windows) == 0 and not self.fully_cleaned:
            # Remove this tensor's dependency registrations from its args
            for arg_id, arg_window in zip(self.args, self.args_windows):
                upstream_tensor = self._store.get_tensor(arg_id)
                upstream_tensor._full_cleanup()
            # No kwargs cleanup
            # Finally clear this tensor's state from the store
            self._store.clear_tensor(self._uuid)

    def to_json(self) -> dict:
        """Serialize this tensor's metadata to JSON."""
        def window_to_dict(w: Optional[TensorWindow]):
            return None if w is None else w.to_dict()

        return {
            'shape': list(self.shape),
            'chunk_size': list(self.chunk_size),
            'dtype': _dtype_to_str(self.dtype),
            'args': [str(a) for a in self.args],
            'args_windows': [window_to_dict(w) for w in self.args_windows],
            'output_window': self.output_window.to_dict(),
            'tile_shape': list(self.tile_shape),
            'tile_bytes': int(self.tile_bytes)
        }
        
