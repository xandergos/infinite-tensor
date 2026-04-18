import torch
import torchvision.transforms.v2.functional as F
import matplotlib.pyplot as plt

from infinite_tensor import InfiniteTensor, TensorWindow

WINDOW_SIZE = 512
KERNEL_SIZE = 5
PADDING = KERNEL_SIZE // 2
PADDED_WINDOW_SIZE = WINDOW_SIZE + PADDING * 2


def base_generator(ctx):
    return torch.rand((3, WINDOW_SIZE, WINDOW_SIZE))


def gaussian_blur(ctx, input_tensor):
    blurred = F.gaussian_blur(input_tensor, kernel_size=KERNEL_SIZE, sigma=1.0)
    return blurred[:, PADDING:-PADDING, PADDING:-PADDING]


padded_input_window = TensorWindow(
    size=(3, PADDED_WINDOW_SIZE, PADDED_WINDOW_SIZE),
    stride=(3, WINDOW_SIZE, WINDOW_SIZE),
    offset=(0, -PADDING, -PADDING),
)

tensor = InfiniteTensor(
    shape=(3, None, None),
    f=base_generator,
    output_window=TensorWindow((3, WINDOW_SIZE, WINDOW_SIZE)),
)

blurred = InfiniteTensor(
    shape=(3, None, None),
    f=gaussian_blur,
    output_window=TensorWindow((3, WINDOW_SIZE, WINDOW_SIZE)),
    args=(tensor,),
    args_windows=[padded_input_window],
)

double_blurred = InfiniteTensor(
    shape=(3, None, None),
    f=gaussian_blur,
    output_window=TensorWindow((3, WINDOW_SIZE, WINDOW_SIZE)),
    args=(blurred,),
    args_windows=[padded_input_window],
)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for ax, title, t in [
    (axs[0], "Original", tensor),
    (axs[1], "Blurred", blurred),
    (axs[2], "Double Blurred", double_blurred),
]:
    ax.imshow(t[:, 500:524, 500:524].permute(1, 2, 0).numpy())
    ax.set_title(title)
plt.show()
