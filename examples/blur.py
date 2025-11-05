import torch
from infinite_tensor import *
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