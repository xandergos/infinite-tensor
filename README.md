# Infinite Tensors

A Python library for performing operations on theoretically infinite tensors using a sliding window approach. This library enables processing of large tensors without loading the entire tensor into memory.

## Installation

## Basic Usage

### Creating an Infinite Tensor

An infinite tensor is created by specifying:

*   Shape (using `None` for infinite dimensions)
*   Function that generates the tensor data in windows
*   The window configuration for processing (shape, stride, offset)

### Creating Dependent Tensors

You can create new tensors that depend on existing ones. The library handles dependencies and memory management automatically: When a tile of a tensor is not needed anymore, it is automatically deleted. To indicate that a tensor is no longer needed by anything but another infinite tensors, you can call tensor.mark_for_cleanup().
When a tile is no longer needed by any infinite tensors, it is automatically deleted to save memory.

## Examples

### Clamping random numbers
```python
def random_generator(ctx):
    # Generate a random 100x100x3 tensor with values between 0 and 0.5
    return torch.rand((3, 100, 100)) * 0.5

def clamp_values(ctx, input_tensor):
    # Simply multiply each value by 2
    return torch.clamp(input_tensor, 0.25, 0.75)

# Create base random tensor
base_tensor = InfiniteTensor(
    shape=(3, None, None),  # 3 channels, infinite height and width
    f=random_generator,
    output_window=TensorWindow((3, 100, 100))
)

# Create doubled tensor
doubled_tensor = InfiniteTensor(
    shape=(3, None, None),
    f=clamp_values,
    output_window=TensorWindow((3, 100, 100)),
    args=(base_tensor,),
    args_windows=[TensorWindow((3, 100, 100))]  # Input window same as output
)

# Visualize the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.imshow(base_tensor[:, :100, :100].permute(1, 2, 0).numpy())
ax1.set_title('Original Random Values (0-0.5)')

ax2.imshow(doubled_tensor[:, :100, :100].permute(1, 2, 0).numpy())
ax2.set_title('Clamped Values (0.25-0.75)')

plt.show()
```


### Seamless, infinite gaussian blur
```python
from infinite_tensors.infinite_tensors import *
import matplotlib.pyplot as plt
import torchvision.transforms.v2.functional as F

def base_generator(ctx):
    return torch.rand((3, 512, 512))

def gaussian_blur(ctx, input_tensor):
    kernel_size = 5
    padding = kernel_size // 2
    
    # Apply blur only to valid region
    blurred = F.gaussian_blur(
        input_tensor,
        kernel_size=kernel_size,
        sigma=1.0
    )
    
    # Calculate valid region (exclude padding that would depend on missing data)
    blurred = blurred[
        :,  # Keep all channels
        padding:-padding if padding > 0 else None,  # Trim vertical padding
        padding:-padding if padding > 0 else None   # Trim horizontal padding
    ]
    
    return blurred

# Usage with adjusted window sizes to account for padding
kernel_size = 5
padding = kernel_size // 2
input_window_size = 512 + padding * 2  # Increase window size to account for padding

tensor = InfiniteTensor(
    shape=(3, None, None),
    f=base_generator,
    output_window=TensorWindow((3, 512, 512))
)

blurred = InfiniteTensor(
    shape=(3, None, None),
    f=gaussian_blur,
    output_window=TensorWindow((3, 512, 512)),  # Original size without padding
    args=(tensor,),
    args_windows=[TensorWindow((3, input_window_size, input_window_size), (3, 512, 512), (0, -padding, -padding))]  # Padded input window
)

double_blurred = InfiniteTensor(
    shape=(3, None, None),
    f=gaussian_blur, 
    output_window=TensorWindow((3, 512, 512)),
    args=(blurred,),
    args_windows=[TensorWindow((3, input_window_size, input_window_size), (3, 512, 512), (0, -padding, -padding))]
)


fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(tensor[:, 500:524, 500:524].permute(1, 2, 0).numpy())
axs[0].set_title('Original')

axs[1].imshow(blurred[:, 500:524, 500:524].permute(1, 2, 0).numpy())
axs[1].set_title('Blurred')

axs[2].imshow(double_blurred[:, 500:524, 500:524].permute(1, 2, 0).numpy())
axs[2].set_title('Double Blurred')

plt.show()
```

## Implementation Details

The library works by:

*   Dividing infinite dimensions into chunks (tiles)
*   Processing data in windows using the provided function
*   Managing dependencies between tensors
*   Automatically cleaning up unused tiles to conserve memory

For the full implementation details, see the source code.

## License

MIT License - See LICENSE file for details.