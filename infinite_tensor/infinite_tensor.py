from __future__ import annotations

import itertools
import logging
import uuid
from collections.abc import Iterable
from typing import Any, Callable

import torch

from infinite_tensor.tensor_window import TensorWindow
from infinite_tensor.tilestore import TileStore
from infinite_tensor.utils import standardize_indices

# COORDINATE SYSTEM DEFINITIONS:
# pixel space - Raw tensor coordinates (what users pass to __getitem__).
# window space - Each point is a window of pixels (sliding-window index for f).

DEFAULT_DTYPE = torch.float32
DEFAULT_DEVICE = torch.device("cpu")

# ERROR MESSAGES
TILE_DELETED_ERROR_MSG = "Tile has been deleted. This indicates either a bug or an attempt to access a tensor after cleanup."
SHAPE_MISMATCH_ERROR_MSG = "Value shape {actual} does not match indexed shape {expected}"
OUTPUT_SHAPE_ERROR_MSG = (
    "Function output shape {actual} does not match expected window shape {expected}"
)
DEVICE_MISMATCH_ERROR_MSG = (
    "Function output is on device {actual}, tensor declared device {expected}."
)

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


class DeviceMismatchError(InfiniteTensorError):
    """Raised when a tensor produced by ``f`` is on a different device than declared."""

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


def _device_to_str(device: torch.device) -> str:
    """Serialize a torch.device to a canonical string (e.g., 'cpu', 'cuda:0')."""
    return str(device)


def _str_to_device(name: str) -> torch.device:
    """Deserialize a device string back to torch.device."""
    return torch.device(name)


def _normalize_device(device: torch.device | str) -> torch.device:
    """Resolve ``device`` to the concrete device PyTorch actually places tensors on.

    ``torch.device("cuda")`` and ``torch.device("cuda:0")`` compare unequal, but
    ``torch.ones((), device="cuda").device`` is ``cuda:0``. We normalize up front
    so the output-device check in ``_process_windows`` stays a plain ``==``.
    """
    resolved = device if isinstance(device, torch.device) else torch.device(device)
    if resolved.type == "cpu" or resolved.index is not None:
        return resolved
    return torch.zeros((), device=resolved).device


def _parse_to_args(*args, **kwargs) -> tuple[torch.device | None, torch.dtype | None]:
    """Parse ``torch.Tensor.to``-style arguments into ``(device, dtype)``.

    Accepts any of:
        - ``.to(device)`` where ``device`` is ``torch.device``, ``str``, or an
          ``int`` CUDA index (``.to(0)`` → ``cuda:0``, matching ``torch.Tensor.to``)
        - ``.to(dtype)`` where ``dtype`` is ``torch.dtype``
        - ``.to(other)`` where ``other`` is a ``torch.Tensor`` (copies its ``device`` and ``dtype``)
        - ``.to(device, dtype)`` positional
        - ``.to(device=..., dtype=...)`` keyword

    Returns ``(device_or_None, dtype_or_None)``. Raises ``TypeError`` on duplicates or
    unrecognized arguments.
    """
    device: Any = kwargs.pop("device", None)
    dtype: torch.dtype | None = kwargs.pop("dtype", None)
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments to .to(): {list(kwargs)}")

    def _is_device_like(value: Any) -> bool:
        if isinstance(value, bool):
            return False
        return isinstance(value, (torch.device, str, int))

    if device is not None and not _is_device_like(device):
        raise TypeError(f"device must be torch.device, str, or int, got {type(device)}")
    if dtype is not None and not isinstance(dtype, torch.dtype):
        raise TypeError(f"dtype must be torch.dtype, got {type(dtype)}")

    for arg in args:
        if isinstance(arg, torch.Tensor):
            if device is not None or dtype is not None:
                raise TypeError(".to(tensor) cannot be combined with device/dtype arguments")
            device = arg.device
            dtype = arg.dtype
        elif isinstance(arg, torch.dtype):
            if dtype is not None:
                raise TypeError("dtype specified more than once to .to()")
            dtype = arg
        elif _is_device_like(arg):
            if device is not None:
                raise TypeError("device specified more than once to .to()")
            device = arg
        else:
            raise TypeError(f"Unsupported argument to .to(): {arg!r}")

    if isinstance(device, int):
        device = torch.device("cuda", device)
    elif isinstance(device, str):
        device = torch.device(device)
    return device, dtype


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
        args: tuple[Any, ...] | None = None,
        args_windows=None,
        dtype: torch.dtype = DEFAULT_DTYPE,
        device: torch.device | str = DEFAULT_DEVICE,
        tile_store: TileStore | None = None,
        tensor_id: Any | None = None,
        batch_size: int | None = None,
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
            device: Device on which ``f`` must return its outputs and on which
                :meth:`__getitem__` will return. Defaults to CPU. ``f`` is
                responsible for transferring upstream arg slices onto this
                device if they live elsewhere.
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

        _validate_tensor_windows(shape, output_window, normalized_args, normalized_args_windows)

        for i, arg in enumerate(normalized_args):
            if not isinstance(arg, InfiniteTensor):
                raise ValidationError("All positional args must be InfiniteTensor instances")
            assert normalized_args_windows[i] is not None, (
                f"Argument window must be provided for infinite tensors (arg {i})"
            )

        self._shape = shape
        self._dtype = dtype
        self._device = _normalize_device(device)
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
    def device(self) -> torch.device:
        return self._device

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
    def batch_size(self) -> int | None:
        return self._batch_size

    def clear_cache(self) -> None:
        """Drop regeneratable cached state for this tensor in the backing store."""
        self._store.clear_cache(self._uuid)

    def to(self, *args, **kwargs) -> InfiniteTensor:
        """Move this tensor's device and/or dtype in place, torch-style.

        Accepts any of ``.to(device)``, ``.to(dtype)``, ``.to(other_tensor)``,
        ``.to(device, dtype)``, or keyword forms (``device=``, ``dtype=``).

        The tensor's declared device/dtype is updated immediately; the backing
        :class:`TileStore` is then asked to migrate any cached state via
        :meth:`TileStore.migrate`. Stores that cannot support the migration
        (e.g. persistent stores on a dtype change) raise; this tensor's
        declared device/dtype is rolled back before the exception propagates.
        Returns ``self`` in all success paths.
        """
        target_device, target_dtype = _parse_to_args(*args, **kwargs)
        new_device = _normalize_device(target_device) if target_device is not None else self._device
        new_dtype = target_dtype if target_dtype is not None else self._dtype
        if new_device == self._device and new_dtype == self._dtype:
            return self
        old_device, old_dtype = self._device, self._dtype
        self._device, self._dtype = new_device, new_dtype
        try:
            self._store.migrate(self._uuid, old_device, old_dtype)
        except Exception:
            self._device, self._dtype = old_device, old_dtype
            raise
        return self

    def __getitem__(self, indices: tuple[int | slice, ...]) -> torch.Tensor:
        """Return tensor values for the requested pixel-space indices."""
        pixel_slices, collapse_dims = standardize_indices(self.shape, indices)
        pixel_slices_tuple: tuple[slice, ...] = tuple(pixel_slices)
        logger.debug(f"Accessing tensor slice with indices: {pixel_slices_tuple}")

        self._store.begin_access(self._uuid)
        try:
            self._ensure_processed_range([pixel_slices_tuple])
            result = self._store.read_pixels(self._uuid, pixel_slices_tuple)
        finally:
            self._store.end_access(self._uuid)

        if collapse_dims:
            target_shape = tuple(s for i, s in enumerate(result.shape) if i not in collapse_dims)
            result = result.reshape(target_shape)
        return result

    def _ensure_processed_range(self, pixel_ranges: Iterable[tuple[slice, ...]]) -> None:
        """Ensure every window intersecting ``pixel_ranges`` is processed in the store."""
        all_indices = itertools.chain.from_iterable(
            self.output_window.intersecting_windows(sub_range, tensor_shape=self.shape)
            for sub_range in pixel_ranges
        )
        self._ensure_processed(all_indices)

    def _ensure_processed(self, window_indices: Iterable[tuple[int, ...]]) -> None:
        """Ensure each listed window is processed: materialize deps, run ``f``, notify store.

        Upstream dependencies are materialized recursively first, then ``f`` is
        invoked (either per-window or in batches) and the result is passed to
        :meth:`TileStore.notify_window_processed`.
        """
        pending_windows: list[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()
        for window_index in window_indices:
            if self._store.is_window_processed(self._uuid, window_index):
                logger.debug(f"Window {window_index} already processed, skipping")
                continue
            if window_index not in seen:
                seen.add(window_index)
                pending_windows.append(window_index)

        for arg in self.args:
            arg._store.begin_access(arg._uuid)
        try:
            for i, arg_window in enumerate(self.args_windows):
                upstream = self.args[i]
                pixel_ranges = [
                    arg_window.get_bounds(window_index) for window_index in pending_windows
                ]
                upstream._ensure_processed_range(pixel_ranges)

            self._process_windows(pending_windows)
        finally:
            for arg in self.args:
                arg._store.end_access(arg._uuid)

    def _process_windows(self, valid_window_indices: list[tuple[int, ...]]) -> None:
        """Invoke ``f`` for each pending window and forward outputs to the store."""
        if self.batch_size is None:
            for window_index in valid_window_indices:
                arg_tensors: list[torch.Tensor] = []
                for i, arg_window in enumerate(self.args_windows):
                    upstream = self.args[i]
                    arg_tensors.append(upstream[arg_window.get_bounds(window_index)])
                with torch.no_grad():
                    output = self.f(window_index, *arg_tensors)

                expected_shape = self.output_window.size
                if tuple(output.shape) != tuple(expected_shape):
                    raise ShapeMismatchError(
                        OUTPUT_SHAPE_ERROR_MSG.format(actual=output.shape, expected=expected_shape)
                    )
                if output.device != self._device:
                    raise DeviceMismatchError(
                        DEVICE_MISMATCH_ERROR_MSG.format(
                            actual=output.device, expected=self._device
                        )
                    )
                self._store.notify_window_processed(self._uuid, window_index, output)
                logger.debug(f"Processed window {window_index}")
        else:

            def apply_batch(batch_window_indices: list[tuple[int, ...]]) -> None:
                arg_tensors: list[list[torch.Tensor]] = [[] for _ in range(len(self.args))]
                for window_index in batch_window_indices:
                    for i, arg_window in enumerate(self.args_windows):
                        upstream = self.args[i]
                        arg_tensors[i].append(upstream[arg_window.get_bounds(window_index)])

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
                    if output.device != self._device:
                        raise DeviceMismatchError(
                            DEVICE_MISMATCH_ERROR_MSG.format(
                                actual=output.device, expected=self._device
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

        def window_to_dict(w: TensorWindow | None):
            return None if w is None else w.to_dict()

        return {
            "shape": list(self.shape),
            "dtype": _dtype_to_str(self.dtype),
            "device": _device_to_str(self.device),
            "args": [a.uuid for a in self.args],
            "args_windows": [window_to_dict(w) for w in self.args_windows],
            "output_window": self.output_window.to_dict(),
        }
