"""Comprehensive tests for InfiniteTensor functionality using pytest."""

import uuid

import pytest
import torch

from infinite_tensor import InfiniteTensor, TensorWindow


class TestInfiniteTensorBasics:
    """Test basic InfiniteTensor functionality."""

    def test_tensor_initialization(self, basic_infinite_tensor):
        """Test that InfiniteTensor initializes correctly."""
        assert basic_infinite_tensor is not None
        assert basic_infinite_tensor.shape == (10, None, None)

    def test_tensor_slicing_shapes(self, basic_infinite_tensor):
        """Test various slicing operations return correct shapes."""
        result = basic_infinite_tensor[0, 0:512, 0:512]
        assert tuple(result.shape) == (512, 512)

        result = basic_infinite_tensor[:, -1024:1024, -1024:1024]
        assert tuple(result.shape) == (10, 2048, 2048)

        result = basic_infinite_tensor[:5, -512:512, -512:512]
        assert tuple(result.shape) == (5, 1024, 1024)

    def test_tensor_values(self, basic_infinite_tensor):
        """Test that tensor returns expected values."""
        result = basic_infinite_tensor[0, 0:512, 0:512]
        expected = torch.ones(512, 512)
        assert torch.allclose(result, expected)


class TestInfiniteTensorDependencies:
    """Test InfiniteTensor dependency functionality."""

    def test_simple_dependency(self, random_infinite_tensor, dependency_func, basic_tensor_window, tile_store):
        """Test basic tensor dependency operations."""
        def dep_func(ctx, base=random_infinite_tensor, w=basic_tensor_window):
            return base[w.get_bounds(ctx)] * 2 - 1
        dep = InfiniteTensor(
            (10, None, None),
            dep_func,
            basic_tensor_window,
            tile_store=tile_store,
            tensor_id=uuid.uuid4(),
        )

        result = dep[:, 0:512, 0:512]
        assert result.max() > 0.99
        assert result.min() < -0.99
        assert abs(result.mean()) < 0.01

    def test_multiple_dependencies(self, zeros_infinite_tensor, increment_func, basic_tensor_window, tile_store):
        """Test chaining multiple dependencies."""
        dep = zeros_infinite_tensor

        for i in range(10):
            def inc_func(ctx, prev=dep, w=basic_tensor_window):
                return prev[w.get_bounds(ctx)] + 1
            dep = InfiniteTensor(
                (10, None, None),
                inc_func,
                basic_tensor_window,
                tile_store=tile_store,
                tensor_id=uuid.uuid4(),
            )

        result = dep[:, 0:512, 0:512]
        expected = torch.full_like(result, 10.0)
        assert torch.allclose(result, expected)

    def test_complex_dependency_with_stride(self, zeros_infinite_tensor, increment_func, strided_tensor_window, tile_store):
        """Test dependency with custom window stride."""
        def inc_stride(ctx, base=zeros_infinite_tensor, w=strided_tensor_window):
            return base[w.get_bounds(ctx)] + 1
        dep = InfiniteTensor(
            (10, None, None),
            inc_stride,
            strided_tensor_window,
            tile_store=tile_store,
            tensor_id=uuid.uuid4(),
        )

        result = dep[:, 0:512, 0:512]
        expected = torch.full_like(result, 4.0)
        assert torch.allclose(result, expected)


class TestInfiniteTensorParametrized:
    """Parametrized tests for various tensor configurations."""

    @pytest.mark.parametrize("slice_config", [
        (slice(0, 1), slice(0, 256), slice(0, 256)),
        (slice(None), slice(100, 612), slice(100, 612)),
        (slice(2, 8), slice(-256, 256), slice(-256, 256)),
    ])
    def test_various_slices(self, basic_infinite_tensor, slice_config):
        """Test various slicing configurations."""
        dim0_slice, dim1_slice, dim2_slice = slice_config
        result = basic_infinite_tensor[dim0_slice, dim1_slice, dim2_slice]

        expected_shape = []
        for i, (slice_obj, orig_dim) in enumerate(zip(slice_config, basic_infinite_tensor.shape)):
            if i == 0:
                start = slice_obj.start or 0
                stop = slice_obj.stop or 10
                expected_shape.append(stop - start)
            else:
                start = slice_obj.start or 0
                stop = slice_obj.stop or 0
                expected_shape.append(stop - start)

        assert tuple(result.shape) == tuple(expected_shape)

    @pytest.mark.parametrize("tensor_shape,window_shape", [
        ((5, None, None), (5, 256, 256)),
        ((20, None, None), (20, 1024, 1024)),
        ((1, None, None), (1, 128, 128)),
    ])
    def test_different_tensor_configurations(self, tensor_shape, window_shape, tile_store):
        """Test different tensor and window shape configurations."""
        def dynamic_tensor_func(ctx):
            return torch.ones(window_shape)

        window = TensorWindow(window_shape)
        tensor = InfiniteTensor(
            tensor_shape,
            dynamic_tensor_func,
            window,
            tile_store=tile_store,
            tensor_id=uuid.uuid4(),
        )

        result = tensor[0, 0:window_shape[1]//2, 0:window_shape[2]//2]
        expected_shape = (window_shape[1]//2, window_shape[2]//2)
        assert tuple(result.shape) == expected_shape


class TestInfiniteTensorEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_slice(self, basic_infinite_tensor):
        """Test empty slice operations."""
        result = basic_infinite_tensor[0, 0:0, 0:0]
        assert tuple(result.shape) == (0, 0)

    def test_negative_indexing(self, basic_infinite_tensor):
        """Test negative indexing works correctly."""
        result = basic_infinite_tensor[:, -100:100, -100:100]
        assert tuple(result.shape) == (10, 200, 200)

        expected = torch.ones(10, 200, 200)
        assert torch.allclose(result, expected)

    def test_bounded_dim_filters_out_of_range_windows(self):
        """Windows whose bounds escape a bounded dim must not be generated."""
        from infinite_tensor.tilestore import MemoryTileStore

        call_count = {"n": 0}

        def counting_func(ctx):
            call_count["n"] += 1
            return torch.ones((7, 64))

        tile_size = 64
        window = TensorWindow(size=(7, tile_size), stride=(7, tile_size))
        tensor = InfiniteTensor(
            (10, None),
            counting_func,
            window,
            tile_store=MemoryTileStore(),
            tensor_id=uuid.uuid4(),
        )

        result = tensor[0:10, 0:tile_size]
        assert tuple(result.shape) == (10, tile_size)
        assert call_count["n"] == 1

    def test_bounded_dim_filters_negative_offset_windows(self):
        """Negative-offset windows that start before a bounded dim are skipped."""
        from infinite_tensor.tilestore import MemoryTileStore

        call_count = {"n": 0}

        def counting_func(ctx):
            call_count["n"] += 1
            return torch.ones((8, 64))

        tile_size = 64
        window = TensorWindow(
            size=(8, tile_size),
            stride=(8, tile_size),
            offset=(-4, 0),
        )
        tensor = InfiniteTensor(
            (8, None),
            counting_func,
            window,
            tile_store=MemoryTileStore(),
            tensor_id=uuid.uuid4(),
        )

        _ = tensor[0:8, 0:tile_size]
        assert call_count["n"] == 0


class TestInfiniteTensorIntegration:
    """Integration tests combining multiple features."""

    def test_multiple_slices_same_tensor(self, basic_infinite_tensor):
        """Test multiple slicing operations on the same tensor."""
        slice1 = basic_infinite_tensor[0:2, 0:100, 0:100]
        slice2 = basic_infinite_tensor[2:4, 100:200, 100:200]
        slice3 = basic_infinite_tensor[:, 50:150, 50:150]

        assert torch.allclose(slice1, torch.ones_like(slice1))
        assert torch.allclose(slice2, torch.ones_like(slice2))
        assert torch.allclose(slice3, torch.ones_like(slice3))

        assert slice1.shape == (2, 100, 100)
        assert slice2.shape == (2, 100, 100)
        assert slice3.shape == (10, 100, 100)

    def test_cache_window_limit_evicts_between_accesses(self):
        """A cache window-count limit evicts oldest entries between accesses."""
        from infinite_tensor.tilestore import MemoryTileStore

        store = MemoryTileStore(cache_size_windows=2)
        call_count = {"n": 0}

        def counting_func(ctx):
            call_count["n"] += 1
            return torch.ones((1, 64, 64))

        tensor = InfiniteTensor(
            (1, None, None),
            counting_func,
            TensorWindow((1, 64, 64)),
            tile_store=store,
            tensor_id=uuid.uuid4(),
        )

        _ = tensor[:, 0:64, 0:64]
        _ = tensor[:, 64:128, 0:64]
        _ = tensor[:, 0:64, 64:128]
        assert call_count["n"] == 3
        assert len(store._windows) == 2

        _ = tensor[:, 0:64, 0:64]
        assert call_count["n"] == 4

    def test_deferred_eviction_during_single_access(self):
        """A single getitem touching more windows than the cache holds must still succeed."""
        from infinite_tensor.tilestore import MemoryTileStore

        store = MemoryTileStore(cache_size_windows=1)

        def ones_func(ctx):
            return torch.ones((1, 64, 64))

        tensor = InfiniteTensor(
            (1, None, None),
            ones_func,
            TensorWindow((1, 64, 64)),
            tile_store=store,
            tensor_id=uuid.uuid4(),
        )

        result = tensor[:, 0:256, 0:256]
        assert result.shape == (1, 256, 256)
        assert torch.allclose(result, torch.ones_like(result))
        assert len(store._windows) == 1

    def test_cache_shared_across_tensors(self):
        """All tensors sharing a MemoryTileStore share its single LRU cache."""
        from infinite_tensor.tilestore import MemoryTileStore

        store = MemoryTileStore(cache_size_windows=3)

        def ones_func(ctx):
            return torch.ones((1, 64, 64))

        def twos_func(ctx):
            return torch.ones((1, 64, 64)) * 2

        uuid_a = uuid.uuid4()
        uuid_b = uuid.uuid4()
        tensor_a = InfiniteTensor(
            (1, None, None),
            ones_func,
            TensorWindow((1, 64, 64)),
            tile_store=store,
            tensor_id=uuid_a,
        )
        tensor_b = InfiniteTensor(
            (1, None, None),
            twos_func,
            TensorWindow((1, 64, 64)),
            tile_store=store,
            tensor_id=uuid_b,
        )

        _ = tensor_a[:, 0:64, 0:64]
        _ = tensor_a[:, 64:128, 0:64]
        _ = tensor_b[:, 0:64, 0:64]
        assert len(store._windows) == 3

        _ = tensor_b[:, 64:128, 0:64]
        assert len(store._windows) == 3
        assert (str(uuid_a), (0, 0, 0)) not in store._windows
        assert (str(uuid_b), (0, 1, 0)) in store._windows

    def test_cache_protected_while_any_tensor_active(self):
        """Eviction must wait until every active begin_access has ended."""
        from infinite_tensor.tilestore import MemoryTileStore

        store = MemoryTileStore(cache_size_windows=1)

        def ones_func(ctx):
            return torch.ones((1, 64, 64))

        tensor_a = InfiniteTensor(
            (1, None, None),
            ones_func,
            TensorWindow((1, 64, 64)),
            tile_store=store,
            tensor_id=uuid.uuid4(),
        )
        tensor_b = InfiniteTensor(
            (1, None, None),
            ones_func,
            TensorWindow((1, 64, 64)),
            tile_store=store,
            tensor_id=uuid.uuid4(),
        )

        store.begin_access(tensor_b.uuid)
        try:
            _ = tensor_a[:, 0:128, 0:64]
            assert len(store._windows) == 2
        finally:
            store.end_access(tensor_b.uuid)
        assert len(store._windows) == 1

    def test_clear_cache_drops_stored_windows(self, tile_store):
        """clear_cache should force re-computation of windows."""
        window = TensorWindow((1, 64, 64))
        call_count = {"n": 0}

        def counting_func(ctx):
            call_count["n"] += 1
            return torch.ones((1, 64, 64))

        tensor = InfiniteTensor(
            (1, None, None),
            counting_func,
            window,
            tile_store=tile_store,
            tensor_id=uuid.uuid4(),
        )

        _ = tensor[:, 0:64, 0:64]
        first_calls = call_count["n"]
        assert first_calls == 1

        _ = tensor[:, 0:64, 0:64]
        assert call_count["n"] == first_calls

        tensor.clear_cache()
        _ = tensor[:, 0:64, 0:64]
        assert call_count["n"] == first_calls + 1
