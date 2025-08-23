"""Utility functions for infinite tensor index processing.

This module contains helper functions for normalizing and standardizing
tensor indices and slices. These functions handle the complex logic of
converting various indexing formats into consistent internal representations.
"""

def normalize_slice(idx: int|slice, shape_dim: int|None):
    """Convert an index or slice to a normalized slice format.
    
    This function ensures all indexing operations use a consistent slice format
    with explicit start, stop, and step values. It handles both integer indices
    (converted to single-element slices) and slice objects.
    
    Args:
        idx: Integer index or slice object to normalize
        shape_dim: Size of the dimension (None for infinite dimensions)
        
    Returns:
        slice: Normalized slice with explicit start, stop, step
        
    Examples:
        normalize_slice(5, None) -> slice(5, 6, 1)  # Single element
        normalize_slice(slice(0, 10, 2), None) -> slice(0, 9, 2)  # Step-aligned
        
    Notes:
        - For slices with steps, adjusts stop to be step-aligned
        - Assumes start/stop are not None (handled by standardize_indices)
        - Critical for consistent indexing across the library
    """
    if isinstance(idx, int):
        return slice(idx, idx + 1, 1)
    else:
        start, stop = idx.start, idx.stop
        if start is None:
            assert shape_dim is not None, "Slice must have start and stop for infinite dimensions"
            start = 0
        if stop is None:
            assert shape_dim is not None, "Slice must have start and stop for infinite dimensions"
            stop = shape_dim
        step = idx.step if idx.step is not None else 1
        return slice(start, start + ((stop - start - 1) // step) * step + 1, step)

def standardize_indices(tensor_shape: tuple[int|None, ...], indices: tuple[int|slice, ...]) -> tuple[slice, ...]:
    """Convert tensor indexing syntax to standardized internal format.
    
    This is the main entry point for processing user-provided tensor indices.
    It handles all the complexity of Python's tensor indexing syntax including
    ellipsis expansion, negative indices, and mixed integer/slice indexing.
    
    Args:
        tensor_shape: Shape of the tensor being indexed (None = infinite dimension)
        indices: User-provided indices (can include ints, slices, ellipsis)
        
    Returns:
        Tuple containing:
        - Normalized slices for all dimensions
        - List of dimension indices that should be collapsed (had integer indices)
        
    Processing Steps:
        1. Convert single indices to tuples
        2. Expand ellipsis (...) to appropriate number of slice(None)
        3. Pad with full slices if too few indices provided
        4. Handle negative indices for finite dimensions
        5. Normalize all indices to slice objects
        6. Track which dimensions should be squeezed
        
    Examples:
        standardize_indices((None, 10), (slice(0, 5), 3))
        -> ([slice(0, 5), slice(3, 4)], [1])  # Dimension 1 should be collapsed
        
        standardize_indices((None, None), (..., slice(0, 10)))
        -> ([slice(None), slice(0, 10)], [])  # Ellipsis expanded
        
    Raises:
        IndexError: If indices are invalid for the tensor shape
    """
    if isinstance(indices, list):
        indices = tuple(indices)
    if not isinstance(indices, tuple):
        indices = (indices,)
        
    # Handle ellipsis by expanding it to the appropriate number of full slices
    if Ellipsis in indices:
        assert indices.count(Ellipsis) == 1, "Only one ellipsis is allowed"
        ellipsis_idx = indices.index(Ellipsis)
        # Calculate how many dimensions the ellipsis should represent
        n_missing = len(tensor_shape) - len(indices) + 1
        # Replace ellipsis with equivalent number of slice(None) objects
        expanded_indices = (
            indices[:ellipsis_idx] + 
            (slice(None),) * n_missing +
            indices[ellipsis_idx + 1:]
        )
    else:
        expanded_indices = indices
        
    # Pad with full slices if needed
    if len(expanded_indices) < len(tensor_shape):
        expanded_indices = expanded_indices + (slice(None),) * (len(tensor_shape) - len(expanded_indices))
        
    # Validate indices length matches shape
    if len(expanded_indices) != len(tensor_shape):
        raise IndexError(f"Too many indices for infinite tensor of dimension {len(tensor_shape)}")
    
    # Handle negative indices for non-infinite dimensions
    processed_indices = []
    for idx, shape_dim in zip(expanded_indices, tensor_shape):
        if shape_dim is not None:
            # Finite dimension: handle negative indexing and bounds checking
            if isinstance(idx, int):
                # Convert negative index to positive (e.g., -1 -> shape_dim-1)
                if idx < 0:
                    idx = shape_dim + idx
                # Validate bounds
                if idx < 0 or idx >= shape_dim:
                    raise IndexError(f"Index {idx} is out of bounds for dimension with size {shape_dim}")
            elif isinstance(idx, slice):
                # Convert negative slice components to positive
                start = idx.start if idx.start is None else (shape_dim + idx.start if idx.start < 0 else idx.start)
                stop = idx.stop if idx.stop is None else (shape_dim + idx.stop if idx.stop < 0 else idx.stop)
                idx = slice(start, stop, idx.step)
        # Infinite dimensions: pass through unchanged (no bounds to check)
        processed_indices.append(idx)
    
    # Identify dimensions that should be collapsed (had integer indices)
    collapse_dims = [i for i, idx in enumerate(processed_indices) if isinstance(idx, int)]
    
    # Normalize all indices to slice objects
    normalized_slices = [normalize_slice(idx, shape_dim) for idx, shape_dim in zip(processed_indices, tensor_shape)]
    
    return normalized_slices, collapse_dims