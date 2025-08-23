"""Test utilities and helper functions for infinite tensor testing."""

import torch
import numpy as np
from typing import Tuple, Callable, Optional
from infinite_tensors.infinite_tensors import InfiniteTensor, TensorWindow


def create_test_tensor(
    shape: Tuple[int, ...], 
    func: Callable, 
    window_shape: Optional[Tuple[int, ...]] = None,
    window_stride: Optional[Tuple[int, ...]] = None,
    **kwargs
) -> InfiniteTensor:
    """Create a test InfiniteTensor with sensible defaults.
    
    Args:
        shape: Shape of the infinite tensor
        func: Function to generate tensor values
        window_shape: Shape of the tensor window (defaults to shape if provided)
        window_stride: Stride for the tensor window
        **kwargs: Additional arguments for InfiniteTensor constructor
        
    Returns:
        Configured InfiniteTensor for testing
    """
    if window_shape is None:
        window_shape = shape
    
    window_kwargs = {}
    if window_stride is not None:
        window_kwargs['window_stride'] = window_stride
    
    window = TensorWindow(window_shape, **window_kwargs)
    
    return InfiniteTensor(shape, func, window, **kwargs)


def assert_tensor_properties(
    tensor: torch.Tensor,
    expected_shape: Tuple[int, ...],
    expected_dtype: Optional[torch.dtype] = None,
    expected_min: Optional[float] = None,
    expected_max: Optional[float] = None,
    expected_mean: Optional[float] = None,
    tolerance: float = 1e-6
) -> None:
    """Assert various properties of a tensor.
    
    Args:
        tensor: Tensor to check
        expected_shape: Expected shape of the tensor
        expected_dtype: Expected data type
        expected_min: Expected minimum value (with tolerance)
        expected_max: Expected maximum value (with tolerance)
        expected_mean: Expected mean value (with tolerance)
        tolerance: Tolerance for floating point comparisons
    """
    assert tuple(tensor.shape) == expected_shape, f"Shape mismatch: got {tensor.shape}, expected {expected_shape}"
    
    if expected_dtype is not None:
        assert tensor.dtype == expected_dtype, f"Dtype mismatch: got {tensor.dtype}, expected {expected_dtype}"
    
    if expected_min is not None:
        actual_min = tensor.min().item()
        assert abs(actual_min - expected_min) <= tolerance, f"Min value mismatch: got {actual_min}, expected {expected_min}"
    
    if expected_max is not None:
        actual_max = tensor.max().item()
        assert abs(actual_max - expected_max) <= tolerance, f"Max value mismatch: got {actual_max}, expected {expected_max}"
    
    if expected_mean is not None:
        actual_mean = tensor.mean().item()
        assert abs(actual_mean - expected_mean) <= tolerance, f"Mean value mismatch: got {actual_mean}, expected {expected_mean}"


def create_dependency_chain(
    base_tensor: InfiniteTensor,
    transform_func: Callable,
    window: TensorWindow,
    chain_length: int
) -> InfiniteTensor:
    """Create a chain of dependent tensors.
    
    Args:
        base_tensor: Starting tensor for the chain
        transform_func: Function to apply at each step
        window: Tensor window to use for dependencies
        chain_length: Number of transformations to apply
        
    Returns:
        Final tensor in the dependency chain
    """
    current = base_tensor
    
    for _ in range(chain_length):
        current = InfiniteTensor(
            current.shape,
            transform_func,
            window,
            args=(current,),
            args_windows=(window,)
        )
    
    return current


def generate_test_data(
    shape: Tuple[int, ...],
    data_type: str = "random",
    seed: Optional[int] = None
) -> torch.Tensor:
    """Generate test data with various patterns.
    
    Args:
        shape: Shape of the tensor to generate
        data_type: Type of data to generate ("random", "zeros", "ones", "sequential", "gaussian")
        seed: Random seed for reproducible results
        
    Returns:
        Generated tensor with specified properties
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    if data_type == "random":
        return torch.rand(shape)
    elif data_type == "zeros":
        return torch.zeros(shape)
    elif data_type == "ones":
        return torch.ones(shape)
    elif data_type == "sequential":
        return torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape)
    elif data_type == "gaussian":
        return torch.randn(shape)
    else:
        raise ValueError(f"Unknown data_type: {data_type}")


class TensorValidator:
    """Helper class for validating tensor operations and results."""
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
    
    def validate_slice_result(
        self, 
        tensor: InfiniteTensor, 
        slice_args: Tuple,
        expected_shape: Tuple[int, ...],
        validation_func: Optional[Callable[[torch.Tensor], bool]] = None
    ) -> torch.Tensor:
        """Validate the result of a slicing operation.
        
        Args:
            tensor: InfiniteTensor to slice
            slice_args: Arguments for slicing (e.g., (slice(0,10), slice(0,100), slice(0,100)))
            expected_shape: Expected shape of the result
            validation_func: Optional function to validate tensor contents
            
        Returns:
            The sliced tensor result
        """
        result = tensor[slice_args]
        assert_tensor_properties(result, expected_shape)
        
        if validation_func is not None:
            assert validation_func(result), "Custom validation function failed"
        
        return result
    
    def validate_dependency_result(
        self,
        dependent_tensor: InfiniteTensor,
        slice_args: Tuple,
        expected_transform: Callable[[torch.Tensor], torch.Tensor],
        base_tensor: InfiniteTensor
    ) -> None:
        """Validate that a dependent tensor correctly transforms its input.
        
        Args:
            dependent_tensor: The dependent tensor to validate
            slice_args: Slice arguments to test
            expected_transform: Function that should transform base tensor values
            base_tensor: The base tensor that the dependent tensor depends on
        """
        dep_result = dependent_tensor[slice_args]
        base_result = base_tensor[slice_args]
        expected_result = expected_transform(base_result)
        
        assert torch.allclose(dep_result, expected_result, atol=self.tolerance), \
            "Dependent tensor does not correctly transform input"


def benchmark_tensor_operation(
    operation: Callable[[], torch.Tensor],
    num_iterations: int = 10,
    warmup_iterations: int = 3
) -> dict:
    """Benchmark a tensor operation for performance testing.
    
    Args:
        operation: Function that performs the operation to benchmark
        num_iterations: Number of timing iterations
        warmup_iterations: Number of warmup iterations to ignore
        
    Returns:
        Dictionary with timing statistics
    """
    import time
    
    # Warmup
    for _ in range(warmup_iterations):
        _ = operation()
    
    # Actual timing
    times = []
    for _ in range(num_iterations):
        start_time = time.perf_counter()
        result = operation()
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'total_time': np.sum(times)
    }
