"""Tests for the deprecated ``get_or_create`` / ``clear_direct_caches`` shims.

These exist so pre-rewrite external scripts (e.g. the terrain-diffusion
``world_pipeline.py`` and ``annotated_infinite_panorama.py``) keep working
against the post-rewrite :class:`TileStore` API. Every call path must emit a
:class:`DeprecationWarning` and still behave as the pre-rewrite API did.
"""

import uuid
import warnings

import pytest
import torch

from infinite_tensor import TensorWindow
from infinite_tensor.tilestore import MemoryTileStore

try:
    import h5py  # noqa: F401

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

if HAS_H5PY:
    from infinite_tensor import HDF5TileStore

requires_h5py = pytest.mark.skipif(not HAS_H5PY, reason="h5py not installed")


def _ones_64():
    """Return a deterministic 64x64 ones tensor."""
    return torch.ones((64, 64))


def _build_tensor_via_get_or_create(store, tensor_id, **extra_kwargs):
    """Invoke the deprecated ``get_or_create`` shim and capture its warnings."""

    def f(ctx):
        return _ones_64()

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        tensor = store.get_or_create(
            tensor_id,
            shape=(None, None),
            f=f,
            output_window=TensorWindow(size=(64, 64), stride=(64, 64)),
            **extra_kwargs,
        )
    return tensor, captured


class TestMemoryGetOrCreate:
    """Deprecation surface for ``MemoryTileStore.get_or_create``."""

    def test_get_or_create_emits_warning_and_returns_working_tensor(self):
        store = MemoryTileStore()
        tensor, captured = _build_tensor_via_get_or_create(
            store,
            str(uuid.uuid4()),
            cache_method="direct",
            cache_limit=2 * 1024 * 1024,
            batch_size=None,
        )

        assert any(issubclass(w.category, DeprecationWarning) for w in captured)
        result = tensor[0:128, 0:128]
        assert tuple(result.shape) == (128, 128)
        assert torch.allclose(result, torch.ones(128, 128))

    def test_cache_limit_maps_to_cache_size_bytes(self):
        store = MemoryTileStore()
        tensor, _ = _build_tensor_via_get_or_create(
            store,
            str(uuid.uuid4()),
            cache_method="direct",
            cache_limit=512 * 1024,
        )
        assert store._cache_size_bytes == 512 * 1024
        assert tuple(tensor[0:64, 0:64].shape) == (64, 64)

    def test_tile_size_is_ignored_on_memory_store(self):
        store = MemoryTileStore()
        _, _ = _build_tensor_via_get_or_create(store, str(uuid.uuid4()), tile_size=128)
        assert not hasattr(store, "tile_size")

    def test_idempotent_reregistration_against_shared_tensor_id(self):
        store = MemoryTileStore()
        tensor_id = str(uuid.uuid4())
        first, _ = _build_tensor_via_get_or_create(store, tensor_id, cache_method="indirect")
        second, _ = _build_tensor_via_get_or_create(store, tensor_id, cache_method="indirect")
        _ = first[0:128, 0:128]
        assert torch.allclose(second[0:128, 0:128], torch.ones(128, 128))


class TestMemoryClearDirectCaches:
    """Deprecation surface for ``MemoryTileStore.clear_direct_caches``."""

    def test_clear_direct_caches_drops_windows_and_warns(self):
        store = MemoryTileStore()
        tensor, _ = _build_tensor_via_get_or_create(
            store, str(uuid.uuid4()), cache_method="direct", cache_limit=None
        )
        _ = tensor[0:128, 0:128]
        assert len(store._windows) > 0

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            store.clear_direct_caches()

        assert any(issubclass(w.category, DeprecationWarning) for w in captured)
        assert len(store._windows) == 0
        assert store._bytes == 0

        result = tensor[0:64, 0:64]
        assert torch.allclose(result, torch.ones(64, 64))


@requires_h5py
class TestHDF5BackCompat:
    """Deprecation surface for ``HDF5TileStore``."""

    def test_tile_cache_size_kwarg_still_accepted(self, tmp_path):
        filepath = tmp_path / "legacy.h5"
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            store = HDF5TileStore(str(filepath), mode="w", tile_cache_size=50)
        assert any(
            "tile_cache_size" in str(w.message) and issubclass(w.category, DeprecationWarning)
            for w in captured
        )
        assert store._cache_size_tiles == 50
        store.close()

    def test_legacy_tile_size_maps_to_store_tile_size(self, tmp_path):
        filepath = tmp_path / "legacy_tile_size.h5"
        store = HDF5TileStore(str(filepath), mode="w", tile_size=64)
        _, _ = _build_tensor_via_get_or_create(store, "tile_size_override", tile_size=128)
        assert store.tile_size == 128
        store.close()

    def test_get_or_create_and_clear_direct_caches_on_hdf5(self, tmp_path):
        filepath = tmp_path / "back_compat.h5"
        store = HDF5TileStore(str(filepath), mode="w", tile_cache_size=100)

        tensor, captured = _build_tensor_via_get_or_create(
            store,
            "tensor_a",
            cache_method="direct",
            cache_limit=1024,
            batch_size=None,
        )
        assert any(issubclass(w.category, DeprecationWarning) for w in captured)
        result = tensor[0:128, 0:128]
        assert tuple(result.shape) == (128, 128)
        assert torch.allclose(result, torch.ones(128, 128))

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            store.clear_direct_caches()
        assert any(issubclass(w.category, DeprecationWarning) for w in captured)
        assert len(store._tile_cache) == 0
        assert store._tile_bytes == 0

        result2 = tensor[0:64, 0:64]
        assert torch.allclose(result2, torch.ones(64, 64))
        store.close()
