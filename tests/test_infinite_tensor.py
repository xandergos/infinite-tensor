"""Comprehensive tests for InfiniteTensor functionality using pytest."""

import pytest
import torch
import numpy as np
from infinite_tensors.infinite_tensors import InfiniteTensor, TensorWindow


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
    
    def test_simple_dependency(self, random_infinite_tensor, dependency_func, basic_tensor_window):
        """Test basic tensor dependency operations."""
        # Create dependent tensor
        dep = InfiniteTensor(
            (10, None, None), 
            dependency_func, 
            basic_tensor_window,
            args=(random_infinite_tensor,), 
            args_windows=(basic_tensor_window,)
        )
        
        # Test the dependency transformation
        result = dep[:, 0:512, 0:512]
        assert result.max() > 0.99
        assert result.min() < -0.99
        assert abs(result.mean()) < 0.01
    
    def test_multiple_dependencies(self, zeros_infinite_tensor, increment_func, basic_tensor_window):
        """Test chaining multiple dependencies."""
        dep = zeros_infinite_tensor
        
        # Chain 10 increment operations
        for i in range(10):
            dep = InfiniteTensor(
                (10, None, None), 
                increment_func, 
                basic_tensor_window,
                args=(dep,), 
                args_windows=(basic_tensor_window,)
            )
        
        result = dep[:, 0:512, 0:512]
        expected = torch.full_like(result, 10.0)
        assert torch.allclose(result, expected)
    
    def test_complex_dependency_with_stride(self, zeros_infinite_tensor, increment_func, strided_tensor_window):
        """Test dependency with custom window stride."""
        dep = InfiniteTensor(
            (10, None, None), 
            increment_func, 
            strided_tensor_window,
            args=(zeros_infinite_tensor,), 
            args_windows=(strided_tensor_window,)
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
    def test_different_tensor_configurations(self, tensor_shape, window_shape):
        """Test different tensor and window shape configurations."""
        # Create a function that matches the window shape
        def dynamic_tensor_func(ctx):
            return torch.ones(window_shape)
        
        window = TensorWindow(window_shape)
        tensor = InfiniteTensor(tensor_shape, dynamic_tensor_func, window)
        
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
    
    def test_cleanup_marking(self, basic_infinite_tensor):
        """Test that cleanup marking works without errors."""
        # This should not raise an exception
        basic_infinite_tensor.mark_for_cleanup()
        
        # Tensor should still be usable after marking for cleanup
        result = basic_infinite_tensor[0, 0:10, 0:10]
        assert result.shape == (10, 10)


class TestInfiniteTensorIntegration:
    """Integration tests combining multiple features."""
    
    def test_dependency_chain_with_cleanup(self, zeros_tensor_func, increment_func, basic_tensor_window):
        """Test a complex dependency chain with proper cleanup."""
        base = InfiniteTensor((10, None, None), zeros_tensor_func, basic_tensor_window)
        
        # Create a chain of dependencies
        tensors = [base]
        for i in range(5):
            dep = InfiniteTensor(
                (10, None, None), 
                increment_func, 
                basic_tensor_window,
                args=(tensors[-1],), 
                args_windows=(basic_tensor_window,)
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


class TestTileManagement:
    """Test tile management and reference counting functionality."""
    
    def test_tile_creation_and_reuse(self, basic_infinite_tensor):
        """Test that tiles are created and reused appropriately."""
        # Get initial tile store state
        initial_keys = set(basic_infinite_tensor._tile_store.keys())
        
        # Make a slice that should create new tiles
        _ = basic_infinite_tensor[:, 0:512, 0:512]
        after_first_keys = set(basic_infinite_tensor._tile_store.keys())
        
        # The same slice should reuse tiles
        _ = basic_infinite_tensor[:, 0:512, 0:512]
        after_second_keys = set(basic_infinite_tensor._tile_store.keys())
        
        assert len(after_first_keys) > len(initial_keys), "No new tiles created"
        assert after_first_keys == after_second_keys, "Tiles not reused"
    
    def test_tile_reference_counting(self, zeros_infinite_tensor, increment_func, basic_tensor_window):
        """Test reference counting with dependencies."""
        # Create a chain of dependent tensors
        dep1 = InfiniteTensor(
            zeros_infinite_tensor.shape,
            increment_func,
            basic_tensor_window,
            args=(zeros_infinite_tensor,),
            args_windows=(basic_tensor_window,)
        )
        
        dep2 = InfiniteTensor(
            zeros_infinite_tensor.shape,
            increment_func,
            basic_tensor_window,
            args=(dep1,),
            args_windows=(basic_tensor_window,)
        )
        
        # Record initial tile counts
        initial_tiles = len(zeros_infinite_tensor._tile_store.keys())
        
        # Make a slice that should create tiles in all tensors
        _ = dep2[:, 0:512, 0:512]
        
        # Check tiles were created
        assert len(zeros_infinite_tensor._tile_store.keys()) > initial_tiles
        
        # Mark base tensor for cleanup
        zeros_infinite_tensor.mark_for_cleanup()
        
        # Tiles should still exist as they're needed by dep1
        assert len(zeros_infinite_tensor._tile_store.keys()) > initial_tiles
        
        # Mark dep1 for cleanup
        dep1.mark_for_cleanup()
        
        # Make another slice that should reuse existing tiles
        _ = dep2[:, 0:512, 0:512]
        
        # Base tiles should still exist
        assert len(zeros_infinite_tensor._tile_store.keys()) > 0
    
    def test_tile_cleanup(self, zeros_infinite_tensor, increment_func, basic_tensor_window):
        """Test that tiles are properly cleaned up when no longer needed."""
        dep = InfiniteTensor(
            zeros_infinite_tensor.shape,
            increment_func,
            basic_tensor_window,
            args=(zeros_infinite_tensor,),
            args_windows=(basic_tensor_window,)
        )
        
        # Create some tiles
        _ = dep[:, 0:512, 0:512]
        initial_base_tiles = len(zeros_infinite_tensor._tile_store.keys())
        initial_dep_tiles = len(dep._tile_store.keys())
        
        assert initial_base_tiles > 0, "No base tiles created"
        assert initial_dep_tiles > 0, "No dependent tiles created"
        
        # Mark both tensors for cleanup
        zeros_infinite_tensor.mark_for_cleanup()
        dep.mark_for_cleanup()
        
        # Force cleanup
        dep._full_cleanup()
        zeros_infinite_tensor._full_cleanup()
        
        assert len(dep._tile_store.keys()) == 0, "Dependent tiles not cleaned up"
        assert len(zeros_infinite_tensor._tile_store.keys()) == 0, "Base tiles not cleaned up"
    
    def test_partial_tile_cleanup(self, zeros_infinite_tensor, increment_func, basic_tensor_window):
        """Test that only unused tiles are cleaned up."""
        dep1 = InfiniteTensor(
            zeros_infinite_tensor.shape,
            increment_func,
            basic_tensor_window,
            args=(zeros_infinite_tensor,),
            args_windows=(basic_tensor_window,)
        )
        
        dep2 = InfiniteTensor(
            zeros_infinite_tensor.shape,
            increment_func,
            basic_tensor_window,
            args=(zeros_infinite_tensor,),
            args_windows=(basic_tensor_window,)
        )
        
        # Create tiles that will be used by both dependents
        region1 = dep1[:, 0:512, 0:512]
        _ = dep2[:, 0:512, 0:512]
        
        # Create tiles that will only be used by dep1
        region2 = dep1[:, 512:1024, 512:1024]
        
        initial_base_tiles = len(zeros_infinite_tensor._tile_store.keys())
        
        # Mark dep1 for cleanup
        dep1.mark_for_cleanup()
        dep1._full_cleanup()
        
        # Some base tiles should remain as they're needed by dep2
        remaining_base_tiles = len(zeros_infinite_tensor._tile_store.keys())
        assert 0 < remaining_base_tiles < initial_base_tiles, "Incorrect partial cleanup"
        
        # Verify dep2 can still access its tiles
        result = dep2[:, 0:512, 0:512]
        assert torch.allclose(result, region1)
        
        # But dep1's unique tiles should be gone
        dep1._tile_store = {}  # Clear any remaining references
        zeros_infinite_tensor._full_cleanup()
        
        # The tiles used only by dep1 should be cleaned up
        assert len(zeros_infinite_tensor._tile_store.keys()) < initial_base_tiles


class TestErrorHandling:
    """Test error conditions and validation."""
    
    def test_invalid_window_size(self, basic_tensor_window):
        """Test error handling for mismatched window sizes."""
        def wrong_size_func(ctx):
            # Return tensor with wrong size
            return torch.ones((5, 256, 256))  # Should be (10, 512, 512)
        
        tensor = InfiniteTensor((10, None, None), wrong_size_func, basic_tensor_window)
        
        with pytest.raises(ValueError, match="Window size mismatch"):
            _ = tensor[:, 0:512, 0:512]
    
    def test_invalid_dimension_map(self):
        """Test error handling for invalid dimension maps."""
        # Missing target dimension
        with pytest.raises(ValueError, match="Invalid dimension map"):
            _ = TensorWindow((3, 256, 256), dimension_map={0: 0, 1: 1})  # Missing dim 2
        
        # Invalid source dimension
        with pytest.raises(ValueError, match="Invalid dimension map"):
            _ = TensorWindow((3, 256, 256), dimension_map={0: 0, 1: 1, 2: 3})  # Dim 3 doesn't exist
    
    def test_device_mismatch(self, basic_tensor_window):
        """Test error handling for device mismatches."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        def gpu_func(ctx):
            return torch.ones((10, 512, 512)).cuda()
            
        tensor = InfiniteTensor((10, None, None), gpu_func, basic_tensor_window)
        
        with pytest.raises(ValueError, match="Device mismatch"):
            _ = tensor[:, 0:512, 0:512]
    
    def test_invalid_dependency_windows(self, zeros_infinite_tensor, increment_func, basic_tensor_window):
        """Test error handling for invalid dependency windows."""
        # Wrong number of windows
        with pytest.raises(ValueError, match="Number of windows"):
            _ = InfiniteTensor(
                zeros_infinite_tensor.shape,
                increment_func,
                basic_tensor_window,
                args=(zeros_infinite_tensor,),
                args_windows=()  # Missing window
            )
        
        # Window shape mismatch
        wrong_window = TensorWindow((5, 256, 256))  # Wrong channel dimension
        with pytest.raises(ValueError, match="Window shape mismatch"):
            _ = InfiniteTensor(
                zeros_infinite_tensor.shape,
                increment_func,
                basic_tensor_window,
                args=(zeros_infinite_tensor,),
                args_windows=(wrong_window,)
            )
    
    def test_invalid_slicing(self, basic_infinite_tensor):
        """Test error handling for invalid slicing operations."""
        # Too many dimensions
        with pytest.raises(IndexError):
            _ = basic_infinite_tensor[:, :, :, :]
        
        # Invalid step size
        with pytest.raises(ValueError, match="Step size"):
            _ = basic_infinite_tensor[:, ::2, :]
    
    def test_cleanup_errors(self, zeros_infinite_tensor, increment_func, basic_tensor_window):
        """Test error handling during cleanup operations."""
        dep = InfiniteTensor(
            zeros_infinite_tensor.shape,
            increment_func,
            basic_tensor_window,
            args=(zeros_infinite_tensor,),
            args_windows=(basic_tensor_window,)
        )
        
        # Create some tiles
        _ = dep[:, 0:512, 0:512]
        
        # Mark for cleanup
        dep.mark_for_cleanup()
        
        # Should raise error when trying to create new tiles after cleanup
        with pytest.raises(RuntimeError, match="marked for cleanup"):
            _ = dep[:, 512:1024, 512:1024]
    
    def test_write_protection(self, basic_infinite_tensor):
        """Test error handling for write operations."""
        # Can't write to tensor with dependencies
        dep = InfiniteTensor(
            basic_infinite_tensor.shape,
            lambda ctx, x: x,
            basic_tensor_window,
            args=(basic_infinite_tensor,),
            args_windows=(basic_tensor_window,)
        )
        
        with pytest.raises(RuntimeError, match="has dependencies"):
            dep[:, 0:512, 0:512] = torch.ones(10, 512, 512)
        
        # Can't write after cleanup marking
        basic_infinite_tensor.mark_for_cleanup()
        with pytest.raises(RuntimeError, match="marked for cleanup"):
            basic_infinite_tensor[:, 0:512, 0:512] = torch.ones(10, 512, 512)
    
    def test_validation_errors(self):
        """Test input validation errors."""
        # Invalid shape
        with pytest.raises(ValueError, match="shape"):
            _ = InfiniteTensor((), lambda ctx: torch.ones(1), TensorWindow((1,)))
        
        # Invalid chunk size
        with pytest.raises(ValueError, match="chunk_size"):
            _ = InfiniteTensor(
                (None, None),
                lambda ctx: torch.ones(512, 512),
                TensorWindow((512, 512)),
                chunk_size=(-1, 256)  # Negative chunk size
            )
        
        # Invalid dtype
        with pytest.raises(TypeError, match="dtype"):
            _ = InfiniteTensor(
                (None, None),
                lambda ctx: torch.ones(512, 512),
                TensorWindow((512, 512)),
                dtype="not_a_dtype"
            )


class TestComplexMemoryCleanup:
    """Test memory cleanup with complex dependency chains."""
    
    def test_diamond_dependency_cleanup(self, zeros_infinite_tensor, increment_func, basic_tensor_window):
        """Test cleanup with diamond-shaped dependency pattern.
        
        Dependency graph:
            base
           /    \
        dep1    dep2
           \    /
           final
        """
        base = zeros_infinite_tensor
        
        # Create two parallel paths
        dep1 = InfiniteTensor(
            base.shape,
            increment_func,
            basic_tensor_window,
            args=(base,),
            args_windows=(basic_tensor_window,)
        )
        
        dep2 = InfiniteTensor(
            base.shape,
            increment_func,
            basic_tensor_window,
            args=(base,),
            args_windows=(basic_tensor_window,)
        )
        
        # Combine paths
        def combine_func(ctx, x1, x2):
            return (x1 + x2) / 2
            
        final = InfiniteTensor(
            base.shape,
            combine_func,
            basic_tensor_window,
            args=(dep1, dep2),
            args_windows=(basic_tensor_window, basic_tensor_window)
        )
        
        # Create initial tiles
        _ = final[:, 0:512, 0:512]
        
        # Record initial tile counts
        base_tiles = len(base._tile_store.keys())
        dep1_tiles = len(dep1._tile_store.keys())
        dep2_tiles = len(dep2._tile_store.keys())
        final_tiles = len(final._tile_store.keys())
        
        assert all(count > 0 for count in [base_tiles, dep1_tiles, dep2_tiles, final_tiles])
        
        # Mark intermediate tensors for cleanup
        dep1.mark_for_cleanup()
        dep2.mark_for_cleanup()
        
        # Base tiles should still exist as both paths need them
        assert len(base._tile_store.keys()) == base_tiles
        
        # Cleanup one path
        dep1._full_cleanup()
        
        # Base tiles still needed by dep2
        assert len(base._tile_store.keys()) == base_tiles
        
        # Cleanup other path
        dep2._full_cleanup()
        
        # Now base tiles can be cleaned up
        base._full_cleanup()
        assert len(base._tile_store.keys()) == 0
        
        # Final tensor should still work
        result = final[:, 0:512, 0:512]
        assert result.shape == (10, 512, 512)
        assert torch.allclose(result, torch.ones_like(result))
    
    def test_cyclic_reference_cleanup(self, zeros_infinite_tensor, increment_func, basic_tensor_window):
        """Test cleanup with cyclic reference patterns.
        
        Tests that cleanup works correctly even when tensors form a cycle in their dependency graph.
        The cycle itself isn't computed (would be infinite), but the tensors should cleanup properly.
        """
        base = zeros_infinite_tensor
        
        # Create a cycle of tensors that reference each other
        def cycle_func(ctx, x1, x2):
            # In practice this would never complete due to infinite recursion
            return x1 + x2
        
        cycle1 = InfiniteTensor(
            base.shape,
            cycle_func,
            basic_tensor_window,
            args=(base, None),  # Second arg will be cycle2
            args_windows=(basic_tensor_window, basic_tensor_window)
        )
        
        cycle2 = InfiniteTensor(
            base.shape,
            cycle_func,
            basic_tensor_window,
            args=(base, cycle1),
            args_windows=(basic_tensor_window, basic_tensor_window)
        )
        
        # Complete the cycle
        cycle1._args = (base, cycle2)
        
        # Mark all for cleanup
        base.mark_for_cleanup()
        cycle1.mark_for_cleanup()
        cycle2.mark_for_cleanup()
        
        # Should cleanup without infinite recursion
        base._full_cleanup()
        cycle1._full_cleanup()
        cycle2._full_cleanup()
        
        assert len(base._tile_store.keys()) == 0
        assert len(cycle1._tile_store.keys()) == 0
        assert len(cycle2._tile_store.keys()) == 0
    
    def test_deep_dependency_chain(self, zeros_infinite_tensor, increment_func, basic_tensor_window):
        """Test cleanup with a deep chain of dependencies."""
        current = zeros_infinite_tensor
        tensors = [current]
        
        # Create a deep chain of dependencies
        for i in range(10):  # Create 10 levels
            next_tensor = InfiniteTensor(
                current.shape,
                increment_func,
                basic_tensor_window,
                args=(current,),
                args_windows=(basic_tensor_window,)
            )
            tensors.append(next_tensor)
            current = next_tensor
        
        # Create tiles throughout the chain
        _ = tensors[-1][:, 0:512, 0:512]
        
        # Verify tiles exist at each level
        tile_counts = [len(t._tile_store.keys()) for t in tensors]
        assert all(count > 0 for count in tile_counts)
        
        # Mark all tensors for cleanup from bottom up
        for tensor in reversed(tensors):
            tensor.mark_for_cleanup()
            tensor._full_cleanup()
        
        # Verify all tiles are cleaned up
        final_counts = [len(t._tile_store.keys()) for t in tensors]
        assert all(count == 0 for count in final_counts)
    
    def test_partial_chain_cleanup(self, zeros_infinite_tensor, increment_func, basic_tensor_window):
        """Test cleanup when only part of a dependency chain is marked."""
        # Create a chain: base -> dep1 -> dep2 -> dep3
        base = zeros_infinite_tensor
        dep1 = InfiniteTensor(
            base.shape,
            increment_func,
            basic_tensor_window,
            args=(base,),
            args_windows=(basic_tensor_window,)
        )
        dep2 = InfiniteTensor(
            base.shape,
            increment_func,
            basic_tensor_window,
            args=(dep1,),
            args_windows=(basic_tensor_window,)
        )
        dep3 = InfiniteTensor(
            base.shape,
            increment_func,
            basic_tensor_window,
            args=(dep2,),
            args_windows=(basic_tensor_window,)
        )
        
        # Create tiles
        _ = dep3[:, 0:512, 0:512]
        
        # Record initial tile counts
        initial_counts = {
            'base': len(base._tile_store.keys()),
            'dep1': len(dep1._tile_store.keys()),
            'dep2': len(dep2._tile_store.keys()),
            'dep3': len(dep3._tile_store.keys())
        }
        
        # Mark middle tensors for cleanup
        dep1.mark_for_cleanup()
        dep2.mark_for_cleanup()
        
        # Cleanup middle tensors
        dep1._full_cleanup()
        dep2._full_cleanup()
        
        # Base should keep its tiles as they're needed for reconstruction
        assert len(base._tile_store.keys()) == initial_counts['base']
        
        # dep3 should still work
        result = dep3[:, 0:512, 0:512]
        assert result.shape == (10, 512, 512)
        assert torch.allclose(result, torch.full_like(result, 3.0))  # Each dep adds 1
        
        # Cleanup everything
        base.mark_for_cleanup()
        dep3.mark_for_cleanup()
        base._full_cleanup()
        dep3._full_cleanup()
        
        # Verify all tiles are cleaned up
        assert len(base._tile_store.keys()) == 0
        assert len(dep1._tile_store.keys()) == 0
        assert len(dep2._tile_store.keys()) == 0
        assert len(dep3._tile_store.keys()) == 0


class TestWindowIntersection:
    """Test window intersection and boundary calculations."""
    
    def test_basic_window_intersection(self):
        """Test basic window intersection calculations."""
        window = TensorWindow((3, 256, 256))
        
        # Test exact window boundaries
        low = window.get_lowest_intersection((0, 0, 0))
        high = window.get_highest_intersection((0, 256, 256))
        assert low == (0, 0, 0)
        assert high == (0, 1, 1)  # Next window starts at 256
        
        # Test partial overlap
        low = window.get_lowest_intersection((0, 100, 100))
        high = window.get_highest_intersection((0, 400, 400))
        assert low == (0, 0, 0)
        assert high == (0, 2, 2)  # Need two windows to cover range
    
    def test_strided_window_intersection(self):
        """Test window intersection with custom strides."""
        window = TensorWindow((3, 256, 256), window_stride=(3, 128, 128))
        
        # Test stride-aligned boundaries
        low = window.get_lowest_intersection((0, 0, 0))
        high = window.get_highest_intersection((0, 256, 256))
        assert low == (0, 0, 0)
        assert high == (0, 2, 2)  # Need more windows due to smaller stride
        
        # Test non-aligned boundaries
        low = window.get_lowest_intersection((0, 150, 150))
        high = window.get_highest_intersection((0, 400, 400))
        assert low == (0, 1, 1)  # Start at stride-aligned window before point
        assert high == (0, 4, 4)  # Need more windows to cover range
    
    def test_window_bounds(self):
        """Test window bound calculations."""
        window = TensorWindow((3, 256, 256))
        
        # Test single window bounds
        bounds = window.get_bounds((0, 0, 0))
        assert bounds == (slice(0, 3), slice(0, 256), slice(0, 256))
        
        # Test offset window bounds
        bounds = window.get_bounds((0, 1, 1))
        assert bounds == (slice(0, 3), slice(256, 512), slice(256, 512))
        
        # Test with custom stride
        window = TensorWindow((3, 256, 256), window_stride=(3, 128, 128))
        bounds = window.get_bounds((0, 1, 1))
        assert bounds == (slice(0, 3), slice(128, 384), slice(128, 384))
    
    def test_window_offset(self):
        """Test window calculations with offset."""
        window = TensorWindow((3, 256, 256), window_offset=(0, -64, -64))
        
        # Test intersection with offset
        low = window.get_lowest_intersection((0, 0, 0))
        high = window.get_highest_intersection((0, 256, 256))
        assert low == (0, 0, 0)  # First window starts at -64
        assert high == (0, 2, 2)  # Need extra window due to offset
        
        # Test bounds with offset
        bounds = window.get_bounds((0, 0, 0))
        assert bounds == (slice(0, 3), slice(-64, 192), slice(-64, 192))
        
        bounds = window.get_bounds((0, 1, 1))
        assert bounds == (slice(0, 3), slice(192, 448), slice(192, 448))
    
    def test_pixel_to_window_mapping(self):
        """Test mapping between pixel and window coordinates."""
        window = TensorWindow((3, 256, 256))
        
        # Test basic mapping
        pixel_range = (slice(0, 3), slice(100, 400), slice(100, 400))
        window_range = window.pixel_range_to_window_range(pixel_range)
        assert window_range == (slice(0, 1), slice(0, 2), slice(0, 2))
        
        # Test with stride
        window = TensorWindow((3, 256, 256), window_stride=(3, 128, 128))
        window_range = window.pixel_range_to_window_range(pixel_range)
        assert window_range == (slice(0, 1), slice(0, 3), slice(0, 3))
        
        # Test with offset
        window = TensorWindow((3, 256, 256), window_offset=(0, -64, -64))
        window_range = window.pixel_range_to_window_range(pixel_range)
        assert window_range == (slice(0, 1), slice(1, 3), slice(1, 3))
    
    def test_boundary_conditions(self):
        """Test window calculations at boundaries."""
        window = TensorWindow((3, 256, 256))
        
        # Test at negative coordinates
        low = window.get_lowest_intersection((0, -256, -256))
        high = window.get_highest_intersection((0, 0, 0))
        assert low == (0, -1, -1)
        assert high == (0, 0, 0)
        
        # Test with very large coordinates
        low = window.get_lowest_intersection((0, 1000000, 1000000))
        high = window.get_highest_intersection((0, 1000256, 1000256))
        assert low == (0, 3906, 3906)  # 1000000 // 256
        assert high == (0, 3907, 3907)  # ceil(1000256 / 256)
        
        # Test single-pixel slices
        pixel_range = (slice(0, 1), slice(500, 501), slice(500, 501))
        window_range = window.pixel_range_to_window_range(pixel_range)
        assert window_range == (slice(0, 1), slice(1, 2), slice(1, 2))
    
    def test_window_intersection_properties(self):
        """Test mathematical properties of window intersections."""
        window = TensorWindow((3, 256, 256))
        
        # Test transitivity
        # If point P is in window W1's range and W1 is in W2's range,
        # then P must be in W2's range
        point = (0, 300, 300)
        w1_range = window.pixel_range_to_window_range((slice(0, 1), slice(300, 301), slice(300, 301)))
        w1_bounds = window.get_bounds((0, w1_range[1].start, w1_range[2].start))
        
        assert (point[1] >= w1_bounds[1].start and point[1] < w1_bounds[1].stop)
        assert (point[2] >= w1_bounds[2].start and point[2] < w1_bounds[2].stop)
        
        # Test symmetry
        # Window W1 intersects W2 iff W2 intersects W1
        w1_idx = (0, 1, 1)
        w2_idx = (0, 2, 2)
        w1_bounds = window.get_bounds(w1_idx)
        w2_bounds = window.get_bounds(w2_idx)
        
        w1_intersects_w2 = (
            w1_bounds[1].stop > w2_bounds[1].start and w1_bounds[1].start < w2_bounds[1].stop and
            w1_bounds[2].stop > w2_bounds[2].start and w1_bounds[2].start < w2_bounds[2].stop
        )
        
        w2_intersects_w1 = (
            w2_bounds[1].stop > w1_bounds[1].start and w2_bounds[1].start < w1_bounds[1].stop and
            w2_bounds[2].stop > w1_bounds[2].start and w2_bounds[2].start < w1_bounds[2].stop
        )
        
        assert w1_intersects_w2 == w2_intersects_w1


class TestTensorWindowMapping:
    """Test TensorWindow dimension mapping functionality."""
    
    def test_basic_dimension_map(self):
        """Test basic dimension mapping between tensors."""
        # Create a window with dimension mapping (C, H, W) -> (H, W)
        window = TensorWindow((3, 256, 256), dimension_map={0: None, 1: 0, 2: 1})
        
        # Create source and target tensors
        def source_func(ctx):
            return torch.ones((3, 256, 256))
        
        source = InfiniteTensor((3, None, None), source_func, window)
        
        # Create a dependent tensor that drops the channel dimension
        def map_func(ctx, x):
            return x.mean(dim=0)  # Average across channels
            
        target = InfiniteTensor(
            (None, None),  # 2D output
            map_func,
            TensorWindow((256, 256)),  # 2D window
            args=(source,),
            args_windows=(window,)
        )
        
        # Test the mapping
        result = target[100:356, 100:356]
        assert result.shape == (256, 256)
        assert torch.allclose(result, torch.ones(256, 256))
    
    def test_reorder_dimensions(self):
        """Test reordering dimensions through window mapping."""
        # Create a window that swaps height and width (C, H, W) -> (C, W, H)
        window = TensorWindow((3, 256, 256), dimension_map={0: 0, 1: 2, 2: 1})
        
        def source_func(ctx):
            return torch.arange(3*256*256).reshape(3, 256, 256).float()
            
        source = InfiniteTensor((3, None, None), source_func, window)
        
        # Create a dependent tensor that transposes the spatial dimensions
        def transpose_func(ctx, x):
            return x.permute(0, 2, 1)
            
        target = InfiniteTensor(
            (3, None, None),
            transpose_func,
            TensorWindow((3, 256, 256)),
            args=(source,),
            args_windows=(window,)
        )
        
        # Test the mapping
        src_slice = source[:, 100:200, 150:250]
        tgt_slice = target[:, 150:250, 100:200]
        assert torch.allclose(src_slice.permute(0, 2, 1), tgt_slice)
    
    def test_expand_dimensions(self):
        """Test expanding dimensions through window mapping."""
        # Create a window that expands 2D to 3D (H, W) -> (1, H, W)
        window = TensorWindow((256, 256), dimension_map={0: 1, 1: 2})
        
        def source_func(ctx):
            return torch.ones((256, 256))
            
        source = InfiniteTensor((None, None), source_func, window)
        
        # Create a dependent tensor that adds a channel dimension
        def expand_func(ctx, x):
            return x.unsqueeze(0).repeat(3, 1, 1)
            
        target = InfiniteTensor(
            (3, None, None),
            expand_func,
            TensorWindow((3, 256, 256)),
            args=(source,),
            args_windows=(window,)
        )
        
        # Test the mapping
        result = target[:, 100:356, 100:356]
        assert result.shape == (3, 256, 256)
        assert torch.allclose(result, torch.ones(3, 256, 256))
    
    def test_complex_dimension_map(self):
        """Test complex dimension mapping with multiple transformations."""
        # Create windows for a complex transformation chain
        # (C, H, W) -> (W, C, H) -> (H, W, C)
        window1 = TensorWindow((3, 256, 256), dimension_map={0: 1, 1: 2, 2: 0})
        window2 = TensorWindow((256, 3, 256), dimension_map={0: 2, 1: 0, 2: 1})
        
        def source_func(ctx):
            return torch.arange(3*256*256).reshape(3, 256, 256).float()
            
        source = InfiniteTensor((3, None, None), source_func, window1)
        
        # First transformation
        def transform1(ctx, x):
            return x.permute(2, 0, 1)
            
        middle = InfiniteTensor(
            (None, 3, None),
            transform1,
            TensorWindow((256, 3, 256)),
            args=(source,),
            args_windows=(window1,)
        )
        
        # Second transformation
        def transform2(ctx, x):
            return x.permute(2, 0, 1)
            
        target = InfiniteTensor(
            (None, None, 3),
            transform2,
            TensorWindow((256, 256, 3)),
            args=(middle,),
            args_windows=(window2,)
        )
        
        # Test the complete transformation chain
        src_slice = source[:, 100:200, 150:250]
        final_slice = target[100:200, 150:250, :]
        
        # Verify the transformations worked correctly
        expected = src_slice.permute(2, 0, 1).permute(2, 0, 1)
        assert torch.allclose(final_slice, expected)


if __name__ == "__main__":
    # Allow running tests directly with python
    pytest.main([__file__, "-v"])