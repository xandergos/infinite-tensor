# Infinite Tensor Operations

A Python library for performing operations on tensors with infinite dimensions using a sliding window approach.

## Overview

This library enables easy processing of theoretically infinite tensors by splitting them into chunks. All tensors can have operations applied to them in a sliding window manner, without the need to load the entire tensor into memory. Tensor memory management is handled automatically to ensure memory and disk usage is minimal.

Example usage:

```python
from infinite_tensors.infinite_tensors import InfiniteTensor, TensorWindow
from scipy import stats

# Create an infinite tensor with shape (1, None, None)
# The tensor will, by default, be initialized with a random uniform value between 0 and 1
t = InfiniteTensor((1, None, None),
                   init_fn=lambda tile_index, tile_shape: torch.rand(tile_shape))

# Define a function we will apply to the tensor
def inverse_gaussian_cdf(x: torch.Tensor) -> torch.Tensor:
    """Compute inverse Gaussian CDF (probit function) for given input tensor."""
    x_np = x.detach().numpy()
    result = stats.norm.ppf(x_np)
    return torch.from_numpy(result).to(x.dtype)

# Apply the function to the tensor, resulting in a new infinite tensor
result = InfiniteTensor((1, None, None), # Shape of the output tensor
                        TensorWindow(t, kernel_size=(256, 256), stride=(256, 256)), # Output window
                        inverse_gaussian_cdf, # Function to apply
                        args=[t], # Arguments to pass to the function
                        kwargs={}, # Keyword arguments to pass to the function
                        args_windows=[TensorWindow(t, kernel_size=(256, 256), stride=(256, 256))], # Windows for input infinite tensors arguments
                        kwargs_windows={}) # Windows for input infinite tensors keyword arguments
```