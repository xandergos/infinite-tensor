"""Comprehensive tests for InfiniteTensor functionality using pytest."""

import pytest
import torch
import numpy as np
from infinite_tensor import TensorWindow
from infinite_tensor.tilestore import MemoryTileStore
import uuid


class TestInfiniteTensorBasics:
    """Test basic InfiniteTensor functionality."""
    
    def test_tensor_initialization(self, basic_infinite_tensor):
        """Test that InfiniteTensor initializes correctly."""
        assert basic_infinite_tensor is not None
        assert basic_infinite_tensor.shape == (10, None, None)
    
    def test_tensor_slicing_shapes(self, basic_infinite_tensor):
        """Test various slicing operations return correct shapes."""
        # Test single slice
        result = basic_infinite_tensor[0, 0:512, 0:512]
        assert tuple(result.shape) == (512, 512)
        
        # Test large slice
        result = basic_infinite_tensor[:, -1024:1024, -1024:1024]
        assert tuple(result.shape) == (10, 2048, 2048)
        
        # Test partial slice
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
        # Create dependent tensor; capture upstream tensor via closure
        def dep_func(ctx, base=random_infinite_tensor, w=basic_tensor_window):
            return base[w.get_bounds(ctx)] * 2 - 1
        dep = tile_store.get_or_create(
            uuid.uuid4(),
            (10, None, None),
            dep_func,
            basic_tensor_window,
        )
        
        # Test the dependency transformation
        result = dep[:, 0:512, 0:512]
        assert result.max() > 0.99
        assert result.min() < -0.99
        assert abs(result.mean()) < 0.01
    
    def test_multiple_dependencies(self, zeros_infinite_tensor, increment_func, basic_tensor_window, tile_store):
        """Test chaining multiple dependencies."""
        dep = zeros_infinite_tensor
        
        # Chain 10 increment operations
        for i in range(10):
            def inc_func(ctx, prev=dep, w=basic_tensor_window):
                return prev[w.get_bounds(ctx)] + 1
            dep = tile_store.get_or_create(
                uuid.uuid4(),
                (10, None, None),
                inc_func,
                basic_tensor_window,
            )
        
        result = dep[:, 0:512, 0:512]
        expected = torch.full_like(result, 10.0)
        assert torch.allclose(result, expected)
    
    def test_complex_dependency_with_stride(self, zeros_infinite_tensor, increment_func, strided_tensor_window, tile_store):
        """Test dependency with custom window stride."""
        def inc_stride(ctx, base=zeros_infinite_tensor, w=strided_tensor_window):
            return base[w.get_bounds(ctx)] + 1
        dep = tile_store.get_or_create(
            uuid.uuid4(),
            (10, None, None),
            inc_stride,
            strided_tensor_window,
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
        
        # Calculate expected shape
        expected_shape = []
        for i, (slice_obj, orig_dim) in enumerate(zip(slice_config, basic_infinite_tensor.shape)):
            if i == 0:  # First dimension has known size
                start = slice_obj.start or 0
                stop = slice_obj.stop or 10
                expected_shape.append(stop - start)
            else:  # Other dimensions are infinite
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
        # Create a function that matches the window shape
        def dynamic_tensor_func(ctx):
            return torch.ones(window_shape)
        
        window = TensorWindow(window_shape)
        tensor = tile_store.get_or_create(uuid.uuid4(), tensor_shape, dynamic_tensor_func, window)
        
        # Test basic functionality
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
        
        # Should be all ones
        expected = torch.ones(10, 200, 200)
        assert torch.allclose(result, expected)


class TestInfiniteTensorIntegration:
    """Integration tests combining multiple features."""

    def test_multiple_slices_same_tensor(self, basic_infinite_tensor):
        """Test multiple slicing operations on the same tensor."""
        slice1 = basic_infinite_tensor[0:2, 0:100, 0:100]
        slice2 = basic_infinite_tensor[2:4, 100:200, 100:200] 
        slice3 = basic_infinite_tensor[:, 50:150, 50:150]
        
        # All should be ones
        assert torch.allclose(slice1, torch.ones_like(slice1))
        assert torch.allclose(slice2, torch.ones_like(slice2))
        assert torch.allclose(slice3, torch.ones_like(slice3))
        
        # Check shapes
        assert slice1.shape == (2, 100, 100)
        assert slice2.shape == (2, 100, 100)
        assert slice3.shape == (10, 100, 100)

    def test_direct_cache_eviction_during_multi_window_access(self, tile_store):
        """Test that direct caching doesn't evict windows needed for current access.
        
        This tests the fix for a bug where cache eviction during _apply_f would
        evict windows before _getitem_direct could read them, causing TileAccessError.
        """
        window = TensorWindow((1, 64, 64))
        
        def ones_func(ctx):
            return torch.ones((1, 64, 64))
        
        # Use direct caching with a very small cache limit (smaller than 2 windows)
        # Each window is 1*64*64*4 = 16KB for float32
        # Set limit to 20KB so only ~1 window fits, but we'll access 4 windows
        tensor = tile_store.get_or_create(
            uuid.uuid4(),
            (1, None, None),
            ones_func,
            window,
            cache_method='direct',
            cache_limit=20 * 1024,
        )
        
        # Access a 2x2 grid of windows (128x128 region with 64x64 windows)
        # This requires 4 windows but cache can only hold ~1
        result = tensor[:, 0:128, 0:128]
        
        assert result.shape == (1, 128, 128)
        assert torch.allclose(result, torch.ones_like(result))

    def test_direct_cache_prioritizes_generated_windows(self, tile_store):
        """Test that generated windows have higher cache priority than used windows.
        
        After generation, the cache order should be:
        1. Used (dependency) windows - promoted first
        2. Generated windows - promoted last (highest priority, evicted last)
        """
        window = TensorWindow((1, 64, 64))
        
        def base_func(ctx):
            return torch.ones((1, 64, 64))
        
        # Create base tensor with direct caching
        base = tile_store.get_or_create(
            uuid.uuid4(),
            (1, None, None),
            base_func,
            window,
            cache_method='direct',
            cache_limit=None,  # No limit for this test
        )
        
        # Pre-populate base cache with several windows
        _ = base[:, 0:64, 0:64]    # window (0, 0, 0)
        _ = base[:, 64:128, 0:64]  # window (0, 1, 0)
        _ = base[:, 0:64, 64:128]  # window (0, 0, 1)
        
        # Create dependent tensor
        def dep_func(ctx, upstream=base, w=window):
            return upstream[w.get_bounds(ctx)] * 2
        
        dep = tile_store.get_or_create(
            uuid.uuid4(),
            (1, None, None),
            dep_func,
            window,
            cache_method='direct',
            cache_limit=None,
        )
        
        # Access dependent tensor - this uses window (0, 0, 0) from base
        result = dep[:, 0:64, 0:64]
        
        # Verify result is correct
        assert torch.allclose(result, torch.ones((1, 64, 64)) * 2)
        
        # Check that base's cache has window (0, 0, 0) promoted (it was used)
        # and dep's cache has window (0, 0, 0) at the end (it was generated)
        base_cache = tile_store._window_cache.get(base.uuid)
        dep_cache = tile_store._window_cache.get(dep.uuid)
        
        assert base_cache is not None
        assert dep_cache is not None
        
        # The used window should be at the end of base's cache
        base_keys = list(base_cache.keys())
        assert (0, 0, 0) == base_keys[-1]
        
        # The generated window should be in dep's cache
        assert (0, 0, 0) in dep_cache
