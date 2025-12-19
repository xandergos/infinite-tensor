# Infinite Tensors

A Python library for performing operations on theoretically infinite tensors using a sliding window approach. This library enables processing of large tensors without loading the entire tensor into memory.

## Installation

Install using pip:
```bash
pip install infinite-tensor
```

## What is an Infinite Tensor?

An Infinite Tensor represents an immutable tensor with one or more unbounded dimensions, defined by a deterministic function `f`. Instead of loading all data into memory at once, it:
- Loads only the parts you need, when you need them
- Processes data in manageable windows
- Caches results so repeated reads are instant

Internally, when you index a region, the system identifies which output windows intersect that region, invokes `f` on each, and sums the outputs together. Results are cached, and no data outside the requested region is generated.

## Key Concepts

1. **Windows**: Define how your processing function sees the data
   - Fixed size (e.g., 64x64 pixels)
   - Outputs in overlapping regions are added together
   - Defined by size, stride, and offset

2. **Infinite Tensors**: Immutable tensors with infinite dimensions
   - Some dimensions can be `None` (infinite), others must be finite
   - Defined by a deterministic function `f` operating on the "output window"
   - Can depend on other infinite tensors

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
    # ctx is the window index (e.g., (wy, wx) for 2D). It is NOT pixel coordinates.
    return torch.ones(512, 512)

# Define the output window seen by your function
window = TensorWindow((512, 512))

# Create an infinite tensor (2D infinite)
tensor = tile_store.get_or_create(
    "my_infinite_tensor",
    shape=(None, None),         # None means infinite dimension
    f=your_processing_function, # A function that takes the index of the current output window as input: e.g (0, 0)
    output_window=window,
    tile_size=512,              # internal tile size (optional)
)
```

### 2. Using the Tensor

Just slice it like a normal tensor (computed on-demand)

```python
result = tensor[0:1024, 0:1024]
```

## Advanced Features

### 1. Caching Methods

InfiniteTensor supports two caching strategies via `cache_method`:

- **`cache_method='indirect'`** (default): Window outputs are accumulated into tiles. Best for persistent storage (e.g., HDF5TileStore) since it uses the least disk space.

- **`cache_method='direct'`**: Window outputs are cached directly with LRU eviction. Best when you want to limit memory usage and don't need persistent storage. Use `cache_limit` to set the max cache size in bytes (default: 10MB), or `None` for unlimited. `tile_size` is ignored with this method.

### 2. Dependency Chaining

Create processing pipelines by making one infinite tensor depend on another.

In this case, f is called like `f(ctx, *args_sliced)`, where ctx is the output window index, and `args_sliced` are the upstream tensors (`args`) sliced by `args_windows`.

```python
import torch
from infinite_tensor import TensorWindow, MemoryTileStore

tile_store = MemoryTileStore()

def zeros_tensor_func(ctx):
    return torch.zeros(10, 512, 512)  # (C, H, W)

base_window = TensorWindow((10, 512, 512))
base = tile_store.get_or_create("my_tensor", (10, None, None), zeros_tensor_func, base_window)

# Define an offset window for the dependent tensor
offset_window = TensorWindow((10, 512, 512), offset=(0, -256, -256))

# The function receives the upstream window directly (already sliced)
def inc_func(ctx, prev):
    return prev + 1

dep = tile_store.get_or_create(
    "my_second_tensor",
    (10, None, None),
    inc_func,
    offset_window,
    args=(base,),
    args_windows=(offset_window,),
)

out = dep[:, 0:512, 0:512]  # ones
```

Note: Manually slicing dependencies inside `f` is not recommended, as it prevents the use of batching, and future versions may introduce automatic memory management utilizing this future.

### 3. Batching

Optionally, `f` can take in a list of tensors, instead of one at a time. The *max* size of the list is given by batch_size. Here is the same example as above but with batching:

```python
import torch
from infinite_tensor import TensorWindow, MemoryTileStore

tile_store = MemoryTileStore()

def zeros_tensor_func(ctx):
    return torch.zeros(10, 512, 512)  # (C, H, W)

base_window = TensorWindow((10, 512, 512))
base = tile_store.get_or_create("my_tensor", (10, None, None), zeros_tensor_func, base_window)

# Define an offset window for the dependent tensor
offset_window = TensorWindow((10, 512, 512), offset=(0, -256, -256))

# The function receives the upstream window directly (already sliced)
# now prev is a list of up to 4 tensors
def inc_func(ctx, prev):
    # return a list of the same size
    prev_stack = torch.stack(prev)
    return [p for p in (prev_stack + 1)]

dep = tile_store.get_or_create(
    "my_second_tensor",
    (10, None, None),
    inc_func,
    offset_window,
    args=(base,),
    args_windows=(offset_window,),
    batch_size=4
)

out = dep[:, 0:512, 0:512]  # ones
```

## Important Notes

1. **Deterministic `f`**: Your function must be deterministic—results are cached assuming `f` is pure.
2. **Create via TileStore**: Construct tensors with `tile_store.get_or_create(...)`. Direct construction of `InfiniteTensor` is not supported.
3. **Avoid manual slicing**: Do not manually slice dependencies. Use `args`/`args_windows` so the framework manages slicing and dependencies.
4. **CPU Only**: Outputs and inputs to `f` are always on the CPU. Returning tensors on other devices will raise errors.
5. **Window Size**: Your function must return exactly the size specified in `TensorWindow`.
6. **Finite Dimensions**: Non-infinite dimensions must fit in memory.

## Example

Check out `examples/blur.py` for a complete example showing how to:
- Process images larger than memory
- Handle boundaries correctly
- Chain multiple processing steps

## License

MIT License - See LICENSE file for details.
