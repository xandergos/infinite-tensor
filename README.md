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
- Stores results in a `TileStore` so repeated reads are instant

Internally, when you index a region, the system identifies which output windows intersect that region, invokes `f` on each, hands the results to the `TileStore`, and then asks the store to assemble the requested pixels.

## Key Concepts

1. **Windows**: Define how your processing function sees the data
   - Fixed size (e.g., 64x64 pixels)
   - Outputs in overlapping regions are added together
   - Defined by size, stride, and offset

2. **Infinite Tensors**: Immutable tensors with infinite dimensions
   - Some dimensions can be `None` (infinite), others must be finite
   - Defined by a deterministic function `f` operating on the "output window"
   - Can depend on other infinite tensors

3. **TileStore**: Backend that owns the processed-window set and its storage
   - `MemoryTileStore` — in-memory dict with optional LRU eviction shared across every tensor it backs
   - `HDF5TileStore` — persistent, accumulates window outputs into fixed-size tiles on disk
   - `PersistentTileStore` — abstract base class for tile-accumulating persistent backends; subclass it to plug in a different storage format (TIFF, PNG, etc.)

## Getting Started

### 1. Creating an Infinite Tensor

Construct an `InfiniteTensor` directly and pass it a store. Reusing the same `tensor_id` against the same store reconnects to previously-processed windows; the store validates that the new tensor's metadata matches the old one.

```python
import uuid
import torch
from infinite_tensor import InfiniteTensor, TensorWindow, MemoryTileStore

tile_store = MemoryTileStore()

def your_processing_function(ctx):
    # ctx is the window index (e.g., (wy, wx) for 2D). It is NOT pixel coordinates.
    return torch.ones(512, 512)

window = TensorWindow((512, 512))

tensor = InfiniteTensor(
    shape=(None, None),
    f=your_processing_function,
    output_window=window,
    tile_store=tile_store,
    tensor_id="my_infinite_tensor",
)
```

### 2. Using the Tensor

Just slice it like a normal tensor (computed on-demand)

```python
result = tensor[0:1024, 0:1024]
```

### 3. Cache size and eviction

`MemoryTileStore` holds a single LRU cache shared by every tensor registered against it. Limits are configured on the store:

```python
tile_store = MemoryTileStore(
    cache_size_bytes=200 * 1024 * 1024,  # 200 MB
    cache_size_windows=None,              # no window-count limit
)

tensor = InfiniteTensor(
    shape=(None, None),
    f=your_processing_function,
    output_window=TensorWindow((512, 512)),
    tile_store=tile_store,
    tensor_id="shared",
)
```

Either (or both) of `cache_size_bytes` / `cache_size_windows` may be `None` (default, unbounded on that axis) or a positive int. All tensors that share the store compete for its slots via pure LRU.

Eviction is deferred while any tensor is inside a `__getitem__`: upstream caches are pinned for the duration of the downstream access, so a single operation that temporarily overruns the cache still succeeds. Once the outermost access returns, the cache is trimmed back to its limits.

### 4. Dropping cached state

```python
tensor.clear_cache()
```

Drops every stored window for this tensor so they'll be recomputed on next access. For `MemoryTileStore` this removes the tensor's entries from the shared cache. `HDF5TileStore` is immutable once a window is processed — `clear_cache` is a no-op. To wipe an HDF5 tensor entirely, call `tile_store.clear_tensor(tensor.uuid)`.

## Advanced Features

### 1. Persistent storage with HDF5


```python
from infinite_tensor import HDF5TileStore, InfiniteTensor, TensorWindow

tile_store = HDF5TileStore("tensor_data.h5", tile_size=512)

tensor = InfiniteTensor(
    shape=(None, None),
    f=your_processing_function,
    output_window=TensorWindow((512, 512)),
    tile_store=tile_store,
    tensor_id="my_infinite_tensor",
)
```

`tile_size` controls how HDF5 accumulates overlapping window outputs into on-disk tiles. It may be an int (applied uniformly to every infinite dim) or a tuple sized to the tensor's infinite-dim count. Reopening the same file with the same `tensor_id` must use matching metadata and `tile_size`, or `register_tensor` raises.

### 2. Dependency Chaining

Create processing pipelines by making one infinite tensor depend on another.

In this case, `f` is called like `f(ctx, *args_sliced)`, where `ctx` is the output window index, and `args_sliced` are the upstream tensors (`args`) sliced by `args_windows`.

```python
import torch
from infinite_tensor import InfiniteTensor, TensorWindow, MemoryTileStore

tile_store = MemoryTileStore()

def zeros_tensor_func(ctx):
    return torch.zeros(10, 512, 512)  # (C, H, W)

base_window = TensorWindow((10, 512, 512))
base = InfiniteTensor(
    (10, None, None),
    zeros_tensor_func,
    base_window,
    tile_store=tile_store,
    tensor_id="my_tensor",
)

offset_window = TensorWindow((10, 512, 512), offset=(0, -256, -256))

def inc_func(ctx, prev):
    return prev + 1

dep = InfiniteTensor(
    (10, None, None),
    inc_func,
    offset_window,
    args=(base,),
    args_windows=(offset_window,),
    tile_store=tile_store,
    tensor_id="my_second_tensor",
)

out = dep[:, 0:512, 0:512]  # ones
```

Note: Manually slicing dependencies inside `f` is not recommended, as it prevents the use of batching, and future versions may introduce automatic memory management utilizing this feature.

### 3. Batching

Optionally, `f` can take in a list of tensors, instead of one at a time. The *max* size of the list is given by `batch_size`. Here is the same example as above but with batching:

```python
import torch
from infinite_tensor import InfiniteTensor, TensorWindow, MemoryTileStore

tile_store = MemoryTileStore()

def zeros_tensor_func(ctx):
    return torch.zeros(10, 512, 512)

base_window = TensorWindow((10, 512, 512))
base = InfiniteTensor(
    (10, None, None),
    zeros_tensor_func,
    base_window,
    tile_store=tile_store,
    tensor_id="my_tensor",
)

offset_window = TensorWindow((10, 512, 512), offset=(0, -256, -256))

def inc_func(ctx, prev):
    prev_stack = torch.stack(prev)
    return [p for p in (prev_stack + 1)]

dep = InfiniteTensor(
    (10, None, None),
    inc_func,
    offset_window,
    args=(base,),
    args_windows=(offset_window,),
    tile_store=tile_store,
    tensor_id="my_second_tensor",
    batch_size=4,
)

out = dep[:, 0:512, 0:512]  # ones
```

## Important Notes

1. **Deterministic `f`**: Your function must be deterministic — stored outputs assume `f` is pure.
2. **Reconnecting**: To reuse stored data, construct the tensor again with the same `tensor_id` and matching metadata against the same store.
3. **Avoid manual slicing**: Do not manually slice dependencies. Use `args`/`args_windows` so the framework manages slicing and dependencies.
4. **CPU Only**: Outputs and inputs to `f` are always on the CPU. Returning tensors on other devices will raise errors.
5. **Window Size**: Your function must return exactly the size specified in `TensorWindow`.
6. **Finite Dimensions**: Non-infinite dimensions must fit in memory.
7. **HDF5 is immutable**: Once a window is processed in `HDF5TileStore`, it cannot be un-processed except by wiping the whole tensor via `clear_tensor`.

## Example

Check out `examples/blur.py` for a complete example showing how to:
- Process images larger than memory
- Handle boundaries correctly
- Chain multiple processing steps

## License

MIT License - See LICENSE file for details.
