
from uuid import UUID
import uuid

from infinite_tensors.utils import normalize_slice


class TensorWindow:
    def __init__(self,
                 window_size: tuple[int, ...], 
                 window_stride: tuple[int, ...] = None, 
                 window_offset: tuple[int, ...] = 0):
        """A sliding window that can be used to apply a function to an infinite tensor.
        
        Args:
            window_size (tuple[int, ...]): The size of the window for each dimension.
            window_stride (tuple[int, ...]): The stride between windows for each dimension.
            window_offset (tuple[int, ...], optional): The offset of the window for each dimension. 
                Defaults to 0, where the top-left corner of the window at index (0, 0, ...) is at the origin.
        """
        assert not isinstance(window_size, int), "window_size must be a tuple"
        if isinstance(window_stride, int):
            window_stride = (window_stride,) * len(window_size)
        if isinstance(window_offset, int):
            window_offset = (window_offset,) * len(window_size)
        self.window_size = window_size
        self.window_stride = window_stride or window_size
        self.window_offset = window_offset
        self._uuid = uuid.uuid4()
        
    @property
    def uuid(self) -> UUID:
        return self._uuid
        
    def get_lowest_intersection(self, point: tuple[int|slice, ...]) -> tuple[int, ...]:
        """Returns the lowest window index that intersects with the given point.
        If there is no window that intersects, returns the next window index.
        
        Args:
            point (tuple[int | slice, ...]): The point to check for intersection. Can be a tuple of ints or slices.
        """
        # Calculate window indices that would contain this point
        window_indices = []
        for p, w, s, o in zip(point, self.window_size, self.window_stride, self.window_offset):
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
        """Returns the highest window index that intersects with the given point."""
        # Calculate window indices that would contain this point
        window_indices = []
        for p, w, s, o in zip(point, self.window_size, self.window_stride, self.window_offset):
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
    
    def pixel_range_to_window_range(self, pixel_ranges: tuple[slice, ...]) -> tuple[slice, ...]:
        """Returns the window ranges that intersect with the given pixel ranges. Returns None if there is no intersection.
        Pixel ranges are expected to be infinite pixel ranges."""
        lowest_intersection = self.get_lowest_intersection((p.start for p in pixel_ranges))
        highest_intersection = self.get_highest_intersection((p.stop - 1 for p in pixel_ranges))
        if any(l > h for l, h in zip(lowest_intersection, highest_intersection)):
            return None  # This means one of the pixel ranges is between two windows
        return tuple(slice(l, h + 1) for l, h in zip(lowest_intersection, highest_intersection))
    
    def get_bounds(self, window_slices: tuple[slice, ...]) -> tuple[slice, ...]:
        """Returns the bounds of the given window slices in pixel space."""
        window_slices = [normalize_slice(w, None) for w in window_slices]
        return tuple(slice(w.start * stride + offset, (w.stop - 1) * stride + size + offset)
                     for w, stride, size, offset in zip(window_slices, self.window_stride, self.window_size, self.window_offset))
