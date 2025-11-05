
from uuid import UUID
import uuid

from infinite_tensor.utils import normalize_slice


class TensorWindow:
    """A sliding window specification for processing infinite tensors.
    
    TensorWindow defines how to slice infinite tensors for processing. It supports:
    - Variable window sizes and strides for different processing patterns
    - Offset windows for padding/boundary handling 
    - Dimension mapping for reshaping data during processing
    
    This is a core abstraction that enables efficient sliding window operations
    on infinite tensors without loading entire datasets into memory.
    """
    
    def __init__(self,
                 size: tuple[int, ...],
                 stride: tuple[int, ...] = None,
                 offset: tuple[int, ...] = None,
                 dimension_map: tuple[int | None, ...] = None):
        """Initialize a tensor window specification.
        
        Args:
            size: Size of the window for each dimension. Determines how much
                  data is included in each window slice.
            stride: Stride between consecutive windows. Defaults to size
                    (non-overlapping windows). Use smaller strides for overlapping
                    windows (e.g., for convolution-like operations).
            offset: Starting offset for the first window. Useful for:
                    - Padding: negative offsets to include boundary regions
                    - Alignment: positive offsets to skip initial data
                    Defaults to (0, 0, ...) placing first window at origin.
            dimension_map: Maps input dimensions to output dimensions. Each position
                          represents an output dimension, value indicates which input
                          dimension to use (None = default slice(0,1)).
                          
                Examples:
                - (0, 1, 2): Identity mapping (no reordering)
                - (2, 0, 1): Dimension rotation
                - (None, 1, 0): 3D output from 2D input with added dimension
                - (1, 0): Transpose 2D dimensions
                
        Raises:
            ValueError: If parameter lengths don't match size length
            AssertionError: If size is not a tuple
        """
        assert not isinstance(size, int), "size must be a tuple"

        if isinstance(stride, int):
            stride = (stride,) * len(size)
        if isinstance(offset, int):
            offset = (offset,) * len(size)

        self.size = size
        self.stride = stride or size
        self.offset = offset or (0,) * len(size)
        self.dimension_map = dimension_map
        
        # Verify all window parameters have same length
        if len(self.stride) != len(self.size):
            raise ValueError(f"stride length ({len(self.stride)}) must match size length ({len(self.size)})")
        if len(self.offset) != len(self.size):
            raise ValueError(f"offset length ({len(self.offset)}) must match size length ({len(self.size)})")
        if self.dimension_map is not None and len(self.dimension_map) != len(self.size):
            raise ValueError(f"dimension_map length ({len(self.dimension_map)}) must match size length ({len(self.size)})")

    # --- Serialization helpers ---
    def to_dict(self) -> dict:
        return {
            'size': list(self.size),
            'stride': list(self.stride),
            'offset': list(self.offset),
            'dimension_map': list(self.dimension_map) if self.dimension_map is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TensorWindow":
        return cls(
            size=tuple(int(x) for x in data['size']),
            stride=tuple(int(x) for x in data['stride']) if data.get('stride') is not None else None,
            offset=tuple(int(x) for x in data['offset']) if data.get('offset') is not None else None,
            dimension_map=tuple(None if x is None else int(x) for x in data['dimension_map']) if data.get('dimension_map') is not None else None,
        )
    
    def map_window_slices(self, input_slices: tuple[slice, ...]) -> tuple[slice, ...]:
        """Apply dimension mapping to transform input slices to output slices.
        
        This is a key operation for handling cases where the input and output tensors
        have different dimension arrangements. For example, when adding a batch dimension
        or transposing spatial dimensions.
        
        Args:
            input_slices: Window slices in the input tensor's dimension order
            
        Returns:
            Window slices reordered according to dimension_map. Dimensions mapped
            to None receive a default slice(0, 1) to add singleton dimensions.
            
        Example:
            If dimension_map=(1, 0) and input_slices=(slice(0,10), slice(5,15)),
            returns (slice(5,15), slice(0,10)) - dimensions are swapped.
        """
        if self.dimension_map is None:
            return input_slices
            
        output_slices = []
        for dim_idx in self.dimension_map:
            if dim_idx is None:
                output_slices.append(slice(0, 1))
            else:
                output_slices.append(input_slices[dim_idx])
        
        return tuple(output_slices)
    
    def inverse_map_window_slices(self, output_slices: tuple[slice, ...], input_dims: int) -> tuple[slice, ...]:
        """Apply inverse dimension mapping to transform output slices to input slices.
        
        This reverses the transformation done by map_window_slices, converting slices
        from the output tensor's dimension order back to the input tensor's order.
        Useful for backpropagation or when you need to map results back to inputs.
        
        Args:
            output_slices: Window slices in the output tensor's dimension order
            input_dims: Number of dimensions in the input tensor
            
        Returns:
            Window slices reordered to match the input tensor's dimensions.
            Dimensions that were mapped from None (singleton dims) are omitted.
            Unknown slices are filled with slice(0, 1).
            
        Example:
            If dimension_map=(1, 0) and output_slices=(slice(5,15), slice(0,10)),
            returns (slice(0,10), slice(5,15)) - dimensions are swapped back.
        """
        if self.dimension_map is None:
            return output_slices
            
        # Count non-None entries to determine input dimension count
        input_slices = [slice(None, None)] * input_dims
        
        for output_idx, input_idx in enumerate(self.dimension_map):
            if input_idx is not None:
                input_slices[input_idx] = output_slices[output_idx]
        
        return tuple(input_slices)
        
    def get_lowest_intersection(self, point: tuple[int|slice, ...]) -> tuple[int, ...]:
        """Find the lowest-indexed window that intersects with a given point/region.
        
        This is essential for determining the starting window when processing a region.
        The algorithm finds the first window whose extent overlaps with the given point.
        
        Args:
            point: Point or region to check for intersection. Can mix integers 
                  (single coordinates) and slices (ranges). For slices, uses 
                  the start coordinate.
        
        Returns:
            Tuple of window indices representing the lowest window that intersects
            the given point. If no intersection exists, returns the next window.
            
        Mathematical Details:
            For each dimension, solves: point < window_index * stride + offset + size
            This finds the smallest window_index where the window end is beyond the point.
        """
        # Calculate window indices that would contain this point
        window_indices = []
        for p, w, s, o in zip(point, self.size, self.stride, self.offset):
            if isinstance(p, slice):
                assert p.start is not None, "Slice must have a start"
                p = p.start
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
        
    def get_highest_intersection(self, point: tuple[int|slice, ...]) -> tuple[int, ...]:
        """Find the highest-indexed window that intersects with a given point/region.
        
        This determines the ending window when processing a region. Combined with
        get_lowest_intersection, defines the complete range of windows needed.
        
        Args:
            point: Point or region to check. For slices, uses the stop-1 coordinate
                  (last included point).
        
        Returns:
            Tuple of window indices representing the highest window that intersects
            the given point.
            
        Mathematical Details:
            For each dimension, solves: point >= window_index * stride + offset
            This finds the largest window_index where the window start is at or before the point.
        """
        # Calculate window indices that would contain this point
        window_indices = []
        for p, w, s, o in zip(point, self.size, self.stride, self.offset):
            if isinstance(p, slice):
                # For slices, use stop-1 since that's the last point included
                assert p.stop is not None, "Slice must have a stop"
                p = p.stop - 1
                
            # Find highest window index where window start <= point
            # Window start = index * stride + offset
            # Solve for index: p >= i * s + o
            # i <= (p - o) / s
            # Calculate floor division
            idx = (p - o) // s
            window_indices.append(idx)
            
        return tuple(window_indices)
    
    def pixel_range_to_window_range(self, pixel_ranges: tuple[slice, ...]) -> tuple[slice, ...] | None:
        """Convert pixel coordinate ranges to window index ranges.
        
        This is a critical function for dependency tracking. It determines which
        windows in a dependent tensor need to be processed when a tile in this
        tensor is accessed.
        
        Args:
            pixel_ranges: Tuple of slices defining pixel regions in infinite dimensions
            
        Returns:
            Tuple of slices defining window ranges that intersect the pixel regions,
            or None if there's no intersection (pixel range falls between windows).
            
        Use Cases:
            - Determining which windows to process for a given pixel access
            - Calculating dependencies between tensors with different window patterns
            - Memory management: knowing when tiles can be safely deleted
        """
        lowest_intersection = self.get_lowest_intersection((p.start for p in pixel_ranges))
        highest_intersection = self.get_highest_intersection((p.stop - 1 for p in pixel_ranges))
        if any(l > h for l, h in zip(lowest_intersection, highest_intersection)):
            return None  # This means one of the pixel ranges is between two windows
        return tuple(slice(l, h + 1) for l, h in zip(lowest_intersection, highest_intersection))
    
    def get_bounds(self, window_slices: tuple[slice, ...], map_slices: bool = True) -> tuple[slice, ...]:
        """Convert window indices to pixel-space coordinates.
        
        This is the inverse of pixel_range_to_window_range, converting window
        coordinates back to the pixel regions they represent. Essential for
        slicing tensors during window processing.
        
        Args:
            window_slices: Window coordinate slices to convert
            map_slices: Whether to apply dimension mapping before conversion.
                       Set to False when dimension mapping has already been applied.
                       Should be True when converting from output window slices to input window slices.
        
        Returns:
            Tuple of slices defining the pixel regions covered by the windows
            
        Mathematical Conversion:
            For each dimension: pixel_start = window_index * stride + offset
                               pixel_end = (window_index + 1) * stride + size + offset - stride
            
        Example:
            Window (1, 2) with stride=(10, 10), size=(5, 5), offset=(0, 0)
            becomes pixel bounds (slice(10, 15), slice(20, 25))
        """
        if map_slices:
            window_slices = self.map_window_slices(window_slices)
        window_slices = [normalize_slice(w, None) for w in window_slices]
        bounds = []
        for i, (stride, size, offset) in enumerate(zip(self.stride, self.size, self.offset)):
            w = window_slices[i]
            start = None if w.start is None else w.start * stride + offset
            stop = None if w.stop is None else (w.stop - 1) * stride + size + offset
            bounds.append(slice(start, stop))
        return tuple(bounds)

    # Keep repr readable for debugging/JSON dumps that might include windows
    def __repr__(self) -> str:
        return (f"TensorWindow(size={self.size}, stride={self.stride}, "
                f"offset={self.offset}, map={self.dimension_map})")
