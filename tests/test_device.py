"""Tests for per-tensor device support and the torch-like ``.to()`` method."""

import json
import uuid

import pytest
import torch

from infinite_tensor import (
    DeviceMismatchError,
    InfiniteTensor,
    TensorWindow,
    ValidationError,
)
from infinite_tensor.tilestore import MemoryTileStore


CUDA_AVAILABLE = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")

try:
    import h5py  # noqa: F401
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

if HAS_H5PY:
    from infinite_tensor import HDF5TileStore

requires_h5py = pytest.mark.skipif(not HAS_H5PY, reason="h5py not installed")


def _make_tensor(device, dtype=torch.float32, tile_store=None, tensor_id=None):
    """Build a simple ones-generating InfiniteTensor on ``device``."""
    window = TensorWindow((64, 64))

    def f(ctx):
        return torch.ones((64, 64), dtype=dtype, device=device)

    return InfiniteTensor(
        shape=(None, None),
        f=f,
        output_window=window,
        dtype=dtype,
        device=device,
        tile_store=tile_store,
        tensor_id=tensor_id if tensor_id is not None else str(uuid.uuid4()),
    )


class TestDeviceAttribute:
    """Cover device storage, properties, and serialization."""

    def test_default_device_is_cpu(self, tile_store):
        tensor = _make_tensor(device="cpu", tile_store=tile_store)
        assert tensor.device == torch.device("cpu")

    def test_device_in_to_json(self, tile_store):
        tensor = _make_tensor(device="cpu", tile_store=tile_store)
        serialized = tensor.to_json()
        assert serialized["device"] == "cpu"

    def test_device_string_accepted(self, tile_store):
        tensor = _make_tensor(device="cpu", tile_store=tile_store)
        assert isinstance(tensor.device, torch.device)

    def test_reregistration_with_different_device_raises(self, tile_store):
        tensor_id = str(uuid.uuid4())
        _make_tensor(device="cpu", tile_store=tile_store, tensor_id=tensor_id)
        with pytest.raises(ValidationError):
            InfiniteTensor(
                shape=(None, None),
                f=lambda ctx: torch.ones((64, 64), device="meta"),
                output_window=TensorWindow((64, 64)),
                device="meta",
                tile_store=tile_store,
                tensor_id=tensor_id,
            )


class TestDeviceMismatchError:
    """The tensor raises if ``f`` returns a tensor on the wrong device."""

    def test_mismatch_raises(self, tile_store):
        window = TensorWindow((64, 64))

        def wrong_device(ctx):
            return torch.ones((64, 64), device="meta")

        tensor = InfiniteTensor(
            shape=(None, None),
            f=wrong_device,
            output_window=window,
            device="cpu",
            tile_store=tile_store,
            tensor_id=str(uuid.uuid4()),
        )
        with pytest.raises(DeviceMismatchError):
            _ = tensor[0:64, 0:64]

    def test_mismatch_in_batched_path_raises(self, tile_store):
        window = TensorWindow((64, 64))

        def wrong_device_batched(ctxs):
            return [torch.ones((64, 64), device="meta") for _ in ctxs]

        tensor = InfiniteTensor(
            shape=(None, None),
            f=wrong_device_batched,
            output_window=window,
            device="cpu",
            tile_store=tile_store,
            tensor_id=str(uuid.uuid4()),
            batch_size=2,
        )
        with pytest.raises(DeviceMismatchError):
            _ = tensor[0:128, 0:64]


@requires_cuda
class TestMemoryStoreCuda:
    """Verify MemoryTileStore produces reads on the declared device."""

    def test_read_returns_on_cuda(self):
        store = MemoryTileStore()
        tensor = _make_tensor(device="cuda", tile_store=store)
        result = tensor[0:64, 0:64]
        assert result.device.type == "cuda"
        assert torch.allclose(result, torch.ones(64, 64, device="cuda"))

    def test_empty_read_still_returns_on_cuda(self):
        store = MemoryTileStore()
        tensor = _make_tensor(device="cuda", tile_store=store)
        result = tensor[0:0, 0:0]
        assert result.device.type == "cuda"


@requires_h5py
@requires_cuda
class TestHDF5StoreCuda:
    """HDF5 tiles round-trip transparently through the device."""

    def test_cuda_roundtrip(self, tmp_path):
        filepath = tmp_path / "cuda.h5"
        store = HDF5TileStore(filepath)
        tensor = _make_tensor(device="cuda", tile_store=store, tensor_id="t1")
        result = tensor[0:64, 0:64]
        assert result.device.type == "cuda"
        assert torch.allclose(result, torch.ones(64, 64, device="cuda"))
        store.close()

        store2 = HDF5TileStore(filepath)

        def f(ctx):
            return torch.ones((64, 64), device="cuda")

        tensor2 = InfiniteTensor(
            shape=(None, None),
            f=f,
            output_window=TensorWindow((64, 64)),
            device="cuda",
            tile_store=store2,
            tensor_id="t1",
        )
        result2 = tensor2[0:64, 0:64]
        assert result2.device.type == "cuda"
        assert torch.allclose(result2, torch.ones(64, 64, device="cuda"))
        store2.close()


class TestToSignature:
    """Cover ``.to()`` argument parsing and no-op fast-path."""

    def test_noop_returns_self(self, tile_store):
        tensor = _make_tensor(device="cpu", tile_store=tile_store)
        assert tensor.to() is tensor
        assert tensor.to("cpu") is tensor
        assert tensor.to(torch.float32) is tensor

    def test_unknown_kwarg_raises(self, tile_store):
        tensor = _make_tensor(device="cpu", tile_store=tile_store)
        with pytest.raises(TypeError):
            tensor.to(unexpected="cpu")

    def test_duplicate_device_raises(self, tile_store):
        tensor = _make_tensor(device="cpu", tile_store=tile_store)
        with pytest.raises(TypeError):
            tensor.to("cpu", device="cpu")

    def test_duplicate_dtype_raises(self, tile_store):
        tensor = _make_tensor(device="cpu", tile_store=tile_store)
        with pytest.raises(TypeError):
            tensor.to(torch.float32, dtype=torch.float64)

    def test_to_tensor_combined_args_raises(self, tile_store):
        tensor = _make_tensor(device="cpu", tile_store=tile_store)
        other = torch.zeros(1)
        with pytest.raises(TypeError):
            tensor.to(other, dtype=torch.float64)

    def test_int_positional_parsed_as_cuda_index(self):
        """``.to(0)`` must resolve to ``cuda:0`` like ``torch.Tensor.to`` does."""
        from infinite_tensor.infinite_tensor import _parse_to_args

        device, dtype = _parse_to_args(0)
        assert device == torch.device("cuda", 0)
        assert dtype is None

    def test_int_device_kwarg_parsed_as_cuda_index(self):
        from infinite_tensor.infinite_tensor import _parse_to_args

        device, dtype = _parse_to_args(device=2)
        assert device == torch.device("cuda", 2)
        assert dtype is None

    def test_bool_rejected_as_device(self):
        """``bool`` is an ``int`` subclass; reject it explicitly so ``.to(True)`` fails."""
        from infinite_tensor.infinite_tensor import _parse_to_args

        with pytest.raises(TypeError):
            _parse_to_args(True)
        with pytest.raises(TypeError):
            _parse_to_args(device=False)


class TestToMemoryDtype:
    """Dtype migration on MemoryTileStore casts cached windows in place."""

    def test_dtype_change_casts_cached_window(self, tile_store):
        tensor = _make_tensor(device="cpu", dtype=torch.float32, tile_store=tile_store)
        _ = tensor[0:64, 0:64]
        cache_keys_before = list(tile_store._windows.keys())
        assert cache_keys_before
        cached_before = tile_store._windows[cache_keys_before[0]]
        assert cached_before.dtype == torch.float32

        tensor.to(torch.float64)
        assert tensor.dtype == torch.float64
        cached_after = tile_store._windows[cache_keys_before[0]]
        assert cached_after.dtype == torch.float64

        result = tensor[0:64, 0:64]
        assert result.dtype == torch.float64

    def test_to_updates_bytes_counter(self):
        store = MemoryTileStore()
        tensor = _make_tensor(device="cpu", dtype=torch.float32, tile_store=store)
        _ = tensor[0:64, 0:64]
        bytes_before = store._bytes
        assert bytes_before > 0
        tensor.to(torch.float64)
        assert store._bytes == bytes_before * 2


class TestToMemoryDevice:
    """Device migration on MemoryTileStore moves cached windows without recompute."""

    @requires_cuda
    def test_device_change_moves_cached_window(self):
        store = MemoryTileStore()
        call_count = {"n": 0}

        def f(ctx):
            call_count["n"] += 1
            return torch.ones((64, 64))

        tensor = InfiniteTensor(
            shape=(None, None),
            f=f,
            output_window=TensorWindow((64, 64)),
            device="cpu",
            tile_store=store,
            tensor_id=str(uuid.uuid4()),
        )
        _ = tensor[0:64, 0:64]
        assert call_count["n"] == 1

        def f_cuda(ctx):
            call_count["n"] += 1
            return torch.ones((64, 64), device="cuda")

        tensor._f = f_cuda
        tensor.to("cuda")
        assert tensor.device.type == "cuda"

        result = tensor[0:64, 0:64]
        assert result.device.type == "cuda"
        assert call_count["n"] == 1

    def test_device_noop_same_str(self, tile_store):
        tensor = _make_tensor(device="cpu", tile_store=tile_store)
        _ = tensor[0:64, 0:64]
        tensor.to("cpu")
        assert tensor.device == torch.device("cpu")


class TestToTensorOverload:
    """``.to(other_tensor)`` copies device and dtype from another torch.Tensor."""

    def test_copies_device_and_dtype(self, tile_store):
        tensor = _make_tensor(device="cpu", dtype=torch.float32, tile_store=tile_store)
        _ = tensor[0:64, 0:64]
        other = torch.zeros(1, dtype=torch.float64)
        tensor.to(other)
        assert tensor.dtype == torch.float64
        assert tensor.device == torch.device("cpu")


@requires_h5py
class TestToHDF5:
    """HDF5 rejects dtype changes and rewrites metadata on device changes."""

    def test_dtype_change_raises_and_rolls_back(self, tmp_path):
        filepath = tmp_path / "dtype.h5"
        store = HDF5TileStore(filepath)
        tensor = _make_tensor(device="cpu", dtype=torch.float32, tile_store=store)
        old_dtype = tensor.dtype
        old_device = tensor.device

        with pytest.raises(ValidationError):
            tensor.to(torch.float64)

        assert tensor.dtype == old_dtype
        assert tensor.device == old_device
        store.close()

    @requires_cuda
    def test_device_change_rewrites_metadata_and_moves_tile_cache(self, tmp_path):
        filepath = tmp_path / "device.h5"
        store = HDF5TileStore(filepath)
        tensor = _make_tensor(device="cpu", tile_store=store, tensor_id="t1")
        _ = tensor[0:64, 0:64]
        cached_keys = [k for k in store._tile_cache if k[0] == "t1"]
        assert cached_keys

        def f_cuda(ctx):
            return torch.ones((64, 64), device="cuda")

        tensor._f = f_cuda
        tensor.to("cuda")
        assert tensor.device.type == "cuda"
        moved_keys = [k for k in store._tile_cache if k[0] == "t1"]
        assert moved_keys == cached_keys
        for key in moved_keys:
            assert store._tile_cache[key].device.type == "cuda"

        with h5py.File(filepath, "r") as file_handle:
            stored = json.loads(
                file_handle["tensors/t1/metadata"].attrs["meta"]
            )
        assert stored["device"] == "cuda:0"

        store.close()

    def test_device_change_cpu_to_cpu_noop_rewrites_metadata(self, tmp_path):
        filepath = tmp_path / "noop.h5"
        store = HDF5TileStore(filepath)
        tensor = _make_tensor(device="cpu", tile_store=store, tensor_id="t1")
        _ = tensor[0:64, 0:64]
        returned = tensor.to("cpu")
        assert returned is tensor
        with h5py.File(filepath, "r") as file_handle:
            stored = json.loads(
                file_handle["tensors/t1/metadata"].attrs["meta"]
            )
        assert stored["device"] == "cpu"
        store.close()
