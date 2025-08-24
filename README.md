# Infinite Tensors

A Python library for performing operations on theoretically infinite tensors using a sliding window approach. This library enables processing of large tensors without loading the entire tensor into memory.

## Installation

Install using pip:
```bash
pip install git+https://github.com/xandergos/infinite-tensor.git
```

## What is an Infinite Tensor?

An Infinite Tensor is a powerful tool that lets you work with data that has one or more unbounded (infinite) dimensions. Instead of loading all data into memory at once, it:
- Loads only the parts you need, when you need them
- Processes data in manageable chunks (windows)
- Automatically manages memory by cleaning up unused data

Think of it like a smart window that slides over your data, processing only what's visible through that window at any time.

## Key Concepts

### Windows and Chunks

1. **Windows**: Define how your processing function sees the data
   - Fixed size (e.g., 64x64 pixels for image processing)
   - Can overlap if needed
   - Your function processes one window at a time

2. **Chunks**: How data is stored internally
   - Larger blocks that contain processed results
   - Automatically managed for memory efficiency
   - You don't need to interact with these directly

## Getting Started

### 1. Creating an Infinite Tensor

Always create tensors through a `TileStore`:

```python
import uuid
import torch
from infinite_tensor import TensorWindow, MemoryTileStore

# Create a tile store (in-memory)
tile_store = MemoryTileStore()

# Define how each window is generated; must match the window's shape
def your_processing_function(ctx):
    # ctx is the window index (e.g., (wy, wx) for 2D)
    return torch.ones(512, 512)

# Define the output window seen by your function
window = TensorWindow((512, 512))

# Create an infinite tensor (2D infinite)
tensor = tile_store.get_or_create(
    uuid.uuid4(),
    shape=(None, None),      # None means infinite dimension
    f=your_processing_function,
    output_window=window,
    chunk_size=512,          # internal tile size (optional)
)
```

### 2. Using the Tensor

```python
# Slice it like a normal tensor (computed on-demand)
result = tensor[0:1024, 0:1024]

# Optional: use a context manager to trigger cleanup when done
with tile_store.get_or_create(uuid.uuid4(), (None, None), your_processing_function, window) as t:
    part = t[10:100, 20:200]
```

## Advanced Features

### 1. Dependency Chaining

Create processing pipelines by making one infinite tensor depend on another.

Automatic windowing via `args` and `args_windows`

```python
import uuid
import torch
from infinite_tensor import TensorWindow, MemoryTileStore

tile_store = MemoryTileStore()

def zeros_tensor_func(ctx):
    return torch.zeros(10, 512, 512)  # (C, H, W)

base_window = TensorWindow((10, 512, 512))
base = tile_store.get_or_create(uuid.uuid4(), (10, None, None), zeros_tensor_func, base_window)

# Define an offset window for the dependent tensor
offset_window = TensorWindow((10, 512, 512), window_offset=(0, -256, -256))

# The function receives the upstream window directly (already sliced)
def inc_func(ctx, prev):
    return prev + 1

dep = tile_store.get_or_create(
    uuid.uuid4(),
    (10, None, None),
    inc_func,
    offset_window,
    args=(base,),
    args_windows=(offset_window,),
)

out = dep[:, 0:512, 0:512]  # ones
```

Note: Do not manually slice dependencies (e.g., using `TensorWindow.get_bounds`). Always pass upstream tensors via `args` with matching `args_windows`. Manual slicing is not recommended and can break dependency tracking and memory management.

### 2. Memory Management

- Memory is automatically managed and cached tiles are reused
- Use context managers for scoped cleanup
- Call `tensor.mark_for_cleanup()` when you know a tensor is no longer needed

```python
tensor.mark_for_cleanup()
```

## Important Notes

1. **Create via TileStore**: Construct tensors with `tile_store.get_or_create(...)`. Direct construction of `InfiniteTensor` is not supported.
2. **Avoid manual slicing**: Do not manually slice dependencies. Use `args`/`args_windows` so the framework manages slicing and dependencies.
3. **CPU Only**: All processing happens on CPU. GPU tensors will raise errors.
4. **Window Size**: Your function must return exactly the size specified in `TensorWindow`.
5. **Finite Dimensions**: Non-infinite dimensions must fit in memory.

## Common Patterns

1. **Image Processing**:
```python
import uuid
import torch
from infinite_tensor import TensorWindow, MemoryTileStore

tile_store = MemoryTileStore()

def process_image(ctx):
    # return HxWxC window; adjust as needed
    return torch.randn(64, 64, 3)

image_window = TensorWindow((64, 64, 3))
image_tensor = tile_store.get_or_create(
    uuid.uuid4(),
    shape=(None, None, 3),
    f=process_image,
    output_window=image_window,
)
region = image_tensor[0:512, 0:512, :]
```

2. **Data Streaming**:
```python
import uuid
import torch
from infinite_tensor import TensorWindow, MemoryTileStore

tile_store = MemoryTileStore()
feature_size = 128

def process_stream(ctx):
    return torch.randn(1000, feature_size)

stream_window = TensorWindow((1000, feature_size))
stream_tensor = tile_store.get_or_create(
    uuid.uuid4(),
    shape=(None, feature_size),
    f=process_stream,
    output_window=stream_window,
)
batch = stream_tensor[0:5000, :]
```

## Troubleshooting

Common issues and solutions:

1. **Memory Issues**:
   - Reduce window size
   - Reduce chunk size
   - Use context managers or `.mark_for_cleanup()` for cleanup

2. **Shape Mismatches**:
   - Ensure your function returns exactly the window size
   - Check that window sizes match between dependent tensors

3. **Performance**:
   - Adjust chunk size to balance memory use and processing overhead
   - Consider window overlap requirements carefully

## Examples

Check out `examples/blur.py` for a complete example showing how to:
- Process images larger than memory
- Handle boundaries correctly
- Chain multiple processing steps

## License

MIT License - See LICENSE file for details.