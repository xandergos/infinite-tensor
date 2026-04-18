import logging
from typing import Any, Callable, Iterable, Optional
import itertools
import uuid

import torch

from infinite_tensor.tensor_window import TensorWindow
from infinite_tensor.tilestore import TileStore
from infinite_tensor.utils import standardize_indices


# COORDINATE SYSTEM DEFINITIONS:
# pixel space - Raw tensor coordinates (what users pass to __getitem__).
# window space - Each point is a window of pixels (sliding-window index for f).

DEFAULT_DTYPE = torch.float32

# ERROR MESSAGES
TILE_DELETED_ERROR_MSG = "Tile has been deleted. This indicates either a bug or an attempt to access a tensor after cleanup."
SHAPE_MISMATCH_ERROR_MSG = "Value shape {actual} does not match indexed shape {expected}"
OUTPUT_SHAPE_ERROR_MSG = "Function output shape {actual} does not match expected window shape {expected}"
DEVICE_MISMATCH_ERROR_MSG = "Device mismatch: value is on {actual}, but infinite tensors require CPU tensors."

logger = logging.getLogger(__name__)


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


def _validate_shape(shape: tuple) -> None:
    """Validate tensor shape specification."""
    if not isinstance(shape, tuple):
        raise ValidationError(f"Shape must be a tuple, got {type(shape)}")
    if len(shape) == 0:
        raise ValidationError("Shape cannot be empty")
    for i, dim in enumerate(shape):
        if dim is not None and (not isinstance(dim, int) or dim <= 0):
            raise ValidationError(f"Dimension {i} must be None or positive integer, got {dim}")


def _validate_function(f: Callable) -> None:
    """Validate the generating function."""
    if not callable(f):
        raise ValidationError(f"Function must be callable, got {type(f)}")


def _validate_window_args(args: tuple, args_windows) -> None:
    """Validate window argument consistency for positional args only."""
    if args_windows is not None and len(args_windows) != len(args):
        raise ValidationError(
            f"args_windows length {len(args_windows)} must match args length {len(args)}"
        )


def _validate_tensor_windows(
    tensor_shape: tuple[int | None, ...],
    output_window: TensorWindow,
    arg_tensors: list,
    arg_windows: list,
) -> None:
    """Ensure window dimensionality matches the corresponding tensor dimensionality."""
    if len(output_window.size) != len(tensor_shape):
        raise ValidationError(
            f"output_window has {len(output_window.size)} dims but tensor has {len(tensor_shape)} dims"
        )

    for i, (arg_tensor, arg_window) in enumerate(zip(arg_tensors, arg_windows)):
        if arg_window is None:
            continue
        arg_tensor_shape = arg_tensor.shape
        if arg_tensor_shape is None:
            continue
        if len(arg_window.size) != len(arg_tensor_shape):
            raise ValidationError(
                f"args_windows[{i}] has {len(arg_window.size)} dims but corresponding arg tensor has {len(arg_tensor_shape)} dims"
            )


class InfiniteTensor:
    """A theoretically infinite tensor defined by a function over sliding windows.

    Data is produced in windows by a deterministic function ``f`` and stored by
    a :class:`TileStore` backend. Construct an ``InfiniteTensor`` with a
    ``tensor_id`` to reuse existing stored windows (the store validates that the
    supplied metadata matches any previously-registered tensor with that id).
    """

    def __init__(
        self,
        shape: tuple[int | None, ...],
        f: Callable,
        output_window: TensorWindow,
        args: tuple = None,
        args_windows=None,
        dtype: torch.dtype = DEFAULT_DTYPE,
        tile_store: Optional[TileStore] = None,
        tensor_id: Optional[Any] = None,
        batch_size: Optional[int] = None,
    ):
        """Initialize an ``InfiniteTensor``.

        Args:
            shape: Tensor shape. ``None`` entries mark infinite dimensions.
            f: ``f(ctx, *args)`` returns a tensor whose shape matches
                ``output_window.size`` exactly. ``ctx`` is the window index in
                window space (or, when ``batch_size`` is set, a list of window
                indices).
            output_window: The window specification for this tensor's outputs.
            args: Upstream ``InfiniteTensor`` dependencies, optional.
            args_windows: One ``TensorWindow`` per entry of ``args``, describing
                how much of each upstream to slice for each window.
            dtype: Element dtype. Defaults to ``torch.float32``.
            tile_store: Store backing this tensor. Auto-creates a
                :class:`MemoryTileStore` (unbounded cache) if omitted. Pass a
                store constructed with explicit ``cache_size_*`` to enable
                LRU eviction.
            tensor_id: Identifier used by the store to key this tensor's data.
                Random UUID if ``None``.
            batch_size: If set, ``f`` receives a list of up to ``batch_size``
                window indices per call and must return a list of tensors.
        """
        _validate_shape(shape)
        _validate_function(f)
        _validate_window_args(args or (), args_windows)

        if tile_store is None:
            from infinite_tensor.tilestore import MemoryTileStore
            tile_store = MemoryTileStore()
        self._store = tile_store
        self._uuid = str(tensor_id) if tensor_id is not None else str(uuid.uuid4())

        normalized_args = list(args or [])
        normalized_args_windows = (
            list(args_windows) if args_windows is not None else [None] * len(normalized_args)
        )

        _validate_tensor_windows(
            shape, output_window, normalized_args, normalized_args_windows
        )

        for i, arg in enumerate(normalized_args):
            if not isinstance(arg, InfiniteTensor):
                raise ValidationError("All positional args must be InfiniteTensor instances")
            assert normalized_args_windows[i] is not None, (
                f"Argument window must be provided for infinite tensors (arg {i})"
            )

        self._shape = shape
        self._dtype = dtype
        self._f = f
        self._args = normalized_args
        self._args_windows = normalized_args_windows
        self._output_window = output_window
        self._batch_size = batch_size

        self._store.register_tensor(self)

    @property
    def shape(self) -> tuple[int | None, ...]:
        return self._shape

    @property
    def uuid(self) -> str:
        return self._uuid

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
    def batch_size(self) -> Optional[int]:
        return self._batch_size

    def clear_cache(self) -> None:
        """Drop regeneratable cached state for this tensor in the backing store."""
        self._store.clear_cache(self._uuid)

    def __getitem__(self, indices: tuple[int | slice, ...]) -> torch.Tensor:
        """Return tensor values for the requested pixel-space indices."""
        indices, collapse_dims = standardize_indices(self.shape, indices)
        indices = tuple(indices)
        logger.debug(f"Accessing tensor slice with indices: {indices}")

        self._store.begin_access(self._uuid)
        try:
            self._ensure_processed_range([indices])
            result = self._store.read_pixels(self._uuid, indices)
        finally:
            self._store.end_access(self._uuid)

        if collapse_dims:
            target_shape = tuple(
                s for i, s in enumerate(result.shape) if i not in collapse_dims
            )
            result = result.reshape(target_shape)
        return result

    def _ensure_processed_range(self, pixel_ranges: Iterable[tuple[slice, ...]]) -> None:
        """Ensure every window intersecting ``pixel_ranges`` is processed in the store."""
        all_indices = itertools.chain.from_iterable(
            itertools.product(
                *[
                    range(lo, hi + 1)
                    for lo, hi in zip(
                        self.output_window.get_lowest_intersection(sub_range),
                        self.output_window.get_highest_intersection(sub_range),
                    )
                ]
            )
            for sub_range in pixel_ranges
        )
        self._ensure_processed(all_indices)

    def _ensure_processed(self, window_indices: Iterable[tuple[int, ...]]) -> None:
        """Ensure each listed window is processed: materialize deps, run ``f``, notify store.

        Upstream dependencies are materialized recursively first, then ``f`` is
        invoked (either per-window or in batches) and the result is passed to
        :meth:`TileStore.notify_window_processed`.
        """
        valid_window_indices: set[tuple[int, ...]] = set()
        for window_index in window_indices:
            if self._store.is_window_processed(self._uuid, window_index):
                logger.debug(f"Window {window_index} already processed, skipping")
                continue
            valid_window_indices.add(window_index)
        valid_window_indices = list(valid_window_indices)

        for arg in self.args:
            arg._store.begin_access(arg._uuid)
        try:
            for i, arg_window in enumerate(self.args_windows):
                upstream = self.args[i]
                pixel_ranges = [
                    arg_window.get_bounds(window_index)
                    for window_index in valid_window_indices
                ]
                upstream._ensure_processed_range(pixel_ranges)

            self._process_windows(valid_window_indices)
        finally:
            for arg in self.args:
                arg._store.end_access(arg._uuid)

    def _process_windows(self, valid_window_indices: list[tuple[int, ...]]) -> None:
        """Invoke ``f`` for each pending window and forward outputs to the store."""
        if self.batch_size is None:
            for window_index in valid_window_indices:
                arg_tensors = []
                for i, arg_window in enumerate(self.args_windows):
                    upstream = self.args[i]
                    arg_tensors.append(upstream[arg_window.get_bounds(window_index)])
                with torch.no_grad():
                    output = self.f(window_index, *arg_tensors)

                expected_shape = self.output_window.size
                if tuple(output.shape) != tuple(expected_shape):
                    raise ShapeMismatchError(
                        OUTPUT_SHAPE_ERROR_MSG.format(
                            actual=output.shape, expected=expected_shape
                        )
                    )
                self._store.notify_window_processed(self._uuid, window_index, output)
                logger.debug(f"Processed window {window_index}")
        else:
            def apply_batch(batch_window_indices: list[tuple[int, ...]]) -> None:
                arg_tensors = [[] for _ in range(len(self.args))]
                for window_index in batch_window_indices:
                    for i, arg_window in enumerate(self.args_windows):
                        upstream = self.args[i]
                        arg_tensors[i].append(
                            upstream[arg_window.get_bounds(window_index)]
                        )

                with torch.no_grad():
                    outputs = self.f(batch_window_indices, *arg_tensors)

                for window_index, output in zip(batch_window_indices, outputs):
                    expected_shape = self.output_window.size
                    if tuple(output.shape) != tuple(expected_shape):
                        raise ShapeMismatchError(
                            OUTPUT_SHAPE_ERROR_MSG.format(
                                actual=output.shape, expected=expected_shape
                            )
                        )
                    self._store.notify_window_processed(self._uuid, window_index, output)
                    logger.debug(f"Processed window {window_index}")

            batched: list[tuple[int, ...]] = []
            for window_index in valid_window_indices:
                batched.append(window_index)
                if len(batched) == self.batch_size:
                    apply_batch(batched)
                    batched = []
            if batched:
                apply_batch(batched)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def to_json(self) -> dict:
        """Serialize this tensor's identifying metadata to JSON-safe primitives."""
        def window_to_dict(w: Optional[TensorWindow]):
            return None if w is None else w.to_dict()

        return {
            'shape': list(self.shape),
            'dtype': _dtype_to_str(self.dtype),
            'args': [a.uuid for a in self.args],
            'args_windows': [window_to_dict(w) for w in self.args_windows],
            'output_window': self.output_window.to_dict(),
        }
