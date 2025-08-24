"""Comprehensive tests for InfiniteTensor functionality using pytest."""

import pytest
import torch
import numpy as np
from infinite_tensors.infinite_tensors import InfiniteTensor, TensorWindow
from infinite_tensors.tilestore import MemoryTileStore
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
        zeros_infinite_tensor.mark_for_cleanup()
        
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


class TestInfiniteTensorMemoryManagement:
    """Test memory management and cleanup functionality."""
    def test_pyramid_cleanup(self, tile_store):
        """Test memory management of tensor with offset window."""
        # Create base zeros tensor
        def zeros_tensor_func(ctx):
            return torch.zeros(10, 512, 512)
        basic_tensor_window = TensorWindow((10, 512, 512))
        base = tile_store.get_or_create(uuid.uuid4(), (10, None, None), zeros_tensor_func, basic_tensor_window)
        
        # Create offset window
        offset_window = TensorWindow((10, 512, 512), window_offset=(0, -256, -256))
        
        # Create incremented tensor with offset window
        def inc_func(ctx, prev):
            return prev + 1
            
        inc_tensor = tile_store.get_or_create(
            uuid.uuid4(),
            (10, None, None),
            inc_func,
            offset_window,
            args=(base,),
            args_windows=(offset_window,),
        )
        base.mark_for_cleanup()
        
        # Check values at origin - should see increment
        result = inc_tensor[:, 0:512, 0:512]
        expected = torch.ones_like(result)
        assert torch.allclose(result, expected)

        assert tile_store.get_tile_for(inc_tensor.uuid, (0, 0)) is not None
        assert tile_store.get_tile_for(base.uuid, (0, 0)) is not None
        zero_tiles = list(filter(lambda x: torch.allclose(x.values, torch.tensor(0.0)), tile_store._tile_store.values()))
        assert len(zero_tiles) == 3 * 3

        inc_tensor[:, -512:1024, -512:1024]
        zero_tiles = list(filter(lambda x: torch.allclose(x.values, torch.tensor(0.0)), tile_store._tile_store.values()))
        assert len(zero_tiles) == 5 * 5 - 1


class TestInfiniteTensorIntegration:
    """Integration tests combining multiple features."""
    
    def test_dependency_chain_with_cleanup(self, zeros_tensor_func, increment_func, basic_tensor_window, tile_store):
        """Test a complex dependency chain with proper cleanup."""
        base = tile_store.get_or_create(uuid.uuid4(), (10, None, None), zeros_tensor_func, basic_tensor_window)
        
        # Create a chain of dependencies
        tensors = [base]
        for i in range(5):
            def inc_func(ctx, prev=tensors[-1], w=basic_tensor_window):
                return prev[w.get_bounds(ctx)] + 1
            dep = tile_store.get_or_create(
                uuid.uuid4(),
                (10, None, None),
                inc_func,
                basic_tensor_window,
            )
            tensors.append(dep)
        
        # Mark intermediate tensors for cleanup
        for tensor in tensors[:-1]:
            tensor.mark_for_cleanup()
        
        # Final result should be 5 (5 increments from 0)
        result = tensors[-1][:, 0:100, 0:100]
        expected = torch.full_like(result, 5.0)
        assert torch.allclose(result, expected)
    
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


if __name__ == "__main__":
    # Allow running tests directly with python
    TestInfiniteTensorMemoryManagement().test_pyramid_cleanup(MemoryTileStore())
    