"""Sliding-window specification for infinite tensors."""

import itertools
from typing import Iterator


class TensorWindow:
    """A sliding-window specification.

    A ``TensorWindow`` owns exactly two pieces of math:

    1. Given pixel-space slices, enumerate every window index that intersects
       them (:meth:`intersecting_windows`).
    2. Given a window index, return the pixel-space slices it covers, applying
       any ``dimension_map`` remap (:meth:`get_bounds`).

    The window geometry is ``size`` (per-dim extent), ``stride`` (spacing
    between successive windows, defaults to ``size`` for non-overlapping
    tiling), and ``offset`` (position of window index ``0``, defaults to
    zeros). ``dimension_map`` optionally reorders/injects dims when turning a
    window index into pixel slices; ``None`` entries insert a singleton axis
    and other entries pick which window-space dim feeds that output dim.
    """

    def __init__(
        self,
        size: tuple[int, ...],
        stride: tuple[int, ...] = None,
        offset: tuple[int, ...] = None,
        dimension_map: tuple[int | None, ...] = None,
    ):
        """Build a window spec.

        Args:
            size: Window extent per dimension.
            stride: Step between consecutive windows. Defaults to ``size``
                (non-overlapping).
            offset: Pixel-space position of window index ``0``. Defaults to
                all zeros.
            dimension_map: Per-output-dim mapping applied by
                :meth:`get_bounds`. ``None`` inserts a singleton dim;
                otherwise the value is an index into the window-space tuple.
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

        if len(self.stride) != len(self.size):
            raise ValueError(
                f"stride length ({len(self.stride)}) must match size length ({len(self.size)})"
            )
        if len(self.offset) != len(self.size):
            raise ValueError(
                f"offset length ({len(self.offset)}) must match size length ({len(self.size)})"
            )
        if self.dimension_map is not None and len(self.dimension_map) != len(self.size):
            raise ValueError(
                f"dimension_map length ({len(self.dimension_map)}) must match size length ({len(self.size)})"
            )

    def to_dict(self) -> dict:
        """Serialize to JSON-safe primitives."""
        return {
            'size': list(self.size),
            'stride': list(self.stride),
            'offset': list(self.offset),
            'dimension_map': list(self.dimension_map) if self.dimension_map is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TensorWindow":
        """Inverse of :meth:`to_dict`."""
        return cls(
            size=tuple(int(x) for x in data['size']),
            stride=tuple(int(x) for x in data['stride']) if data.get('stride') is not None else None,
            offset=tuple(int(x) for x in data['offset']) if data.get('offset') is not None else None,
            dimension_map=tuple(None if x is None else int(x) for x in data['dimension_map']) if data.get('dimension_map') is not None else None,
        )

    def intersecting_windows(
        self, pixel_slices: tuple[slice, ...]
    ) -> Iterator[tuple[int, ...]]:
        """Yield every window index whose pixel extent intersects ``pixel_slices``.

        Args:
            pixel_slices: One slice per dimension in window-space order. Each
                slice must have explicit ``start``/``stop``; ``step`` is
                ignored (windows intersect on pixel extent, not stride).

        Yields:
            Window-index tuples ``(w_0, ..., w_n)``. Empty if any dimension's
            intersection is empty.

        For each dim with stride ``s``, offset ``o``, size ``w``:
            ``low = ceil((start - o - w + 1) / s)``
            ``high = floor((stop - 1 - o) / s)``
        """
        per_dim_ranges: list[range] = []
        for pixel_slice, window_size, stride, offset in zip(
            pixel_slices, self.size, self.stride, self.offset
        ):
            numerator = pixel_slice.start - offset - window_size + 1
            if numerator >= 0:
                low = (numerator + stride - 1) // stride
            else:
                low = -(-numerator // stride)
            high = (pixel_slice.stop - 1 - offset) // stride
            if low > high:
                return
            per_dim_ranges.append(range(low, high + 1))
        yield from itertools.product(*per_dim_ranges)

    def get_bounds(self, window_index: tuple[int, ...]) -> tuple[slice, ...]:
        """Return the pixel-space slices covered by a single window.

        Args:
            window_index: One integer per window-space dimension.

        Returns:
            Pixel-space slices, one per output dimension. When
            ``dimension_map`` is set it picks, per output dim, which
            window-space dim feeds that output (``None`` inserts a singleton
            axis). Stride/size/offset are always applied in output-dim order.
            Returned slices have ``step=None``; persistent stores normalize
            this on their side.
        """
        if self.dimension_map is None:
            mapped_coords: tuple[int | None, ...] = tuple(window_index)
        else:
            mapped_coords = tuple(
                None if source_dim is None else window_index[source_dim]
                for source_dim in self.dimension_map
            )

        bounds: list[slice] = []
        for output_dim, window_coord in enumerate(mapped_coords):
            if window_coord is None:
                bounds.append(slice(0, 1))
                continue
            stride = self.stride[output_dim]
            size = self.size[output_dim]
            offset = self.offset[output_dim]
            start = window_coord * stride + offset
            stop = window_coord * stride + size + offset
            bounds.append(slice(start, stop))
        return tuple(bounds)

    def __repr__(self) -> str:
        return (
            f"TensorWindow(size={self.size}, stride={self.stride}, "
            f"offset={self.offset}, map={self.dimension_map})"
        )
