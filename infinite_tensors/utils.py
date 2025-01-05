def normalize_slice(idx: int|slice, shape_dim: int|None):
    """Normalizes a slice so that stop is one greater than the last element. Assumes start/stop is not None."""
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
    """
    Converts indices to a normalized form, with all ellipses expanded and slices normalized.
    Also returns a list of dimensions that should be collapsed because integers were passed.
    """
    if isinstance(indices, list):
        indices = tuple(indices)
    if not isinstance(indices, tuple):
        indices = (indices,)
        
    # Handle ellipsis by expanding it to the appropriate number of full slices
    if Ellipsis in indices:
        assert indices.count(Ellipsis) == 1, "Only one ellipsis is allowed"
        ellipsis_idx = indices.index(Ellipsis)
        n_missing = len(tensor_shape) - len(indices) + 1
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
    
    collapse_dims = [i for i, idx in enumerate(expanded_indices) if isinstance(idx, int)]
    return [normalize_slice(idx, shape_dim) for idx, shape_dim in zip(expanded_indices, tensor_shape)], collapse_dims