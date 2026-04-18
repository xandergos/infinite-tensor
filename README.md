# Infinite Tensors

A Python library for operating on theoretically infinite tensors using a sliding-window approach. Process tensors that do not fit in memory (or that are genuinely unbounded) by computing and caching only the windows you actually read.

## Installation

```bash
pip install infinite-tensor
```

Persistent HDF5 storage is optional:

```bash
pip install "infinite-tensor[hdf5]"
```

## What is an Infinite Tensor?

An `InfiniteTensor` is an immutable tensor defined by a deterministic function `f`. Any subset of its dimensions can be `None` (unbounded); the rest are finite. Fully-finite shapes are legal too, but the interesting case is at least one infinite dim.

When you slice it:

1. The library enumerates every output window that intersects the requested region.
2. `f` is called on each window (and on any upstream dependencies that feed it).
3. Results are handed to a `TileStore`, which caches them so repeat reads are free.
4. The store assembles the requested pixels and returns a `torch.Tensor`.

Overlapping output windows are **summed** in the region they overlap. That is the only blending rule.

## Key Concepts

1. **`TensorWindow`**: output window specification. Fixed `size`, optional `stride` (defaults to `size`, non-overlapping) and `offset` (defaults to zeros). Your `f` must return a tensor of exactly `size`.
2. **`InfiniteTensor`**: immutable tensor with a shape (any number of `None` dims), a deterministic `f`, a declared `dtype` and `device`, and a backing `TileStore`. Can depend on other `InfiniteTensor`s through `args` / `args_windows`.
3. **`TileStore`**: owns the processed-window set and the storage. Pick one:
   - `MemoryTileStore`: in-memory, single LRU cache shared by every tensor registered against it.
   - `HDF5TileStore`: persistent, accumulates window outputs into fixed-size tiles in an HDF5 file on disk. Requires `h5py` (install via the `hdf5` extra).

Subclass `PersistentTileStore` (`from infinite_tensor.tilestore.persistent import PersistentTileStore`) to add other durable backends; see [Extending with a custom persistent backend](#extending-with-a-custom-persistent-backend).

## Getting Started

### 1. Creating a tensor

```python
import torch
from infinite_tensor import InfiniteTensor, TensorWindow, MemoryTileStore

tile_store = MemoryTileStore()

def your_processing_function(ctx):
    # ctx is the window index (e.g. (wy, wx) for 2D), NOT pixel coordinates.
    return torch.ones(512, 512)

tensor = InfiniteTensor(
    shape=(None, None),
    f=your_processing_function,
    output_window=TensorWindow((512, 512)),
    tile_store=tile_store,
    tensor_id="my_infinite_tensor",
)
```

`tile_store` is optional; if omitted, a fresh `MemoryTileStore` with an unbounded cache is created for you.

`tensor_id` defaults to a random UUID4. Pass an explicit id if you want to **reconnect** to stored data on a later run (for persistent stores) or across tensor re-creations in the same session. On re-construction with the same id, the store validates that the new tensor's metadata matches the old one.

### 2. Slicing

Index like a normal tensor. Computation happens on demand:

```python
result = tensor[0:1024, 0:1024]
```

`__getitem__` supports integer indices, slices, and negative indices. Integer indices squeeze the dimension; slices preserve it.

### 3. Device and dtype

Pass `device=` and/or `dtype=` at construction. `f` must return tensors matching both exactly (`DeviceMismatchError` / `DtypeMismatchError` otherwise). Defaults are `cpu` / `torch.float32`. Upstream arg slices are **not** auto-transferred; if an upstream lives elsewhere, move/cast it inside `f`.

`tensor.to(...)` mirrors `torch.Tensor.to` (device, dtype, or another tensor) and migrates cached state in place. `MemoryTileStore` supports both device and dtype changes; `HDF5TileStore` (and any `PersistentTileStore`) supports device changes but raises `ValidationError` on dtype changes, so `clear_tensor` and rebuild if you need to re-cast on disk.

### 4. Cache size and eviction

`MemoryTileStore` holds one LRU cache shared by every tensor registered against it:

```python
tile_store = MemoryTileStore(
    cache_size_bytes=200 * 1024 * 1024,  # 200 MB
    cache_size_windows=None,             # no window-count limit
)
```

Either (or both) of `cache_size_bytes` / `cache_size_windows` may be `None` (default, unbounded on that axis) or a positive int.

Eviction is deferred while any tensor is inside a `__getitem__` call. Upstream caches are also pinned for the duration of a downstream read, so a single operation that temporarily overruns the cache still succeeds. Once the outermost access returns, the cache trims back to its limits.

`HDF5TileStore` has its own separate tile LRU; see [Persistent storage with HDF5](#9-persistent-storage-with-hdf5).

### 5. Clearing state

```python
tensor.clear_cache()
tile_store.clear_tensor(tensor.uuid)
```

- `tensor.clear_cache()` drops regeneratable cached state for this tensor. On `MemoryTileStore` it wipes the tensor's windows so they recompute on next access. On `HDF5TileStore` (and any `PersistentTileStore`) it flushes dirty tiles to disk and then drops the tensor's in-memory tile cache; persisted tiles and the processed-window record are untouched.
- `tile_store.clear_tensor(tensor_id)` wipes **everything** for the tensor: registration, metadata, cached windows, and (on persistent stores) the on-disk tile data and processed-window record. This is the only way to un-process a window on a persistent store.

## Advanced Features

### 6. Dependency chaining

Build pipelines by making one infinite tensor depend on another. `f` is called as `f(ctx, *args_sliced)`, where `args_sliced[i]` is the upstream `args[i]` sliced by `args_windows[i].get_bounds(ctx)`.

Keep the **output window** separate from the **arg window**; the arg window typically carries the padding/offset:

```python
import torch
import torchvision.transforms.v2.functional as F
from infinite_tensor import InfiniteTensor, TensorWindow, MemoryTileStore

tile_store = MemoryTileStore()

WINDOW = 512
PADDING = 2
PADDED = WINDOW + 2 * PADDING

def random_image(ctx):
    return torch.rand(3, WINDOW, WINDOW)

base = InfiniteTensor(
    shape=(3, None, None),
    f=random_image,
    output_window=TensorWindow((3, WINDOW, WINDOW)),
    tile_store=tile_store,
    tensor_id="base",
)

padded_input_window = TensorWindow(
    size=(3, PADDED, PADDED),
    stride=(3, WINDOW, WINDOW),
    offset=(0, -PADDING, -PADDING),
)

def blur(ctx, padded_input):
    blurred = F.gaussian_blur(padded_input, kernel_size=5, sigma=1.0)
    return blurred[:, PADDING:-PADDING, PADDING:-PADDING]

blurred = InfiniteTensor(
    shape=(3, None, None),
    f=blur,
    output_window=TensorWindow((3, WINDOW, WINDOW)),
    args=(base,),
    args_windows=(padded_input_window,),
    tile_store=tile_store,
    tensor_id="blurred",
)

out = blurred[:, 0:1024, 0:1024]
```

Do **not** manually index `base` inside `f`. Use `args` / `args_windows`; the framework manages slicing, batching, and cross-store cache pinning.

### 7. Batching

Setting `batch_size=N` changes the contract of `f`:

- `ctx` becomes a `list[tuple[int, ...]]` of up to `N` window indices.
- Each positional arg becomes a `list[torch.Tensor]` of the same length (one sliced upstream tensor per window in the batch).
- `f` must return a `list[torch.Tensor]` of the same length.

```python
def blur_batched(ctxs, padded_inputs):
    stack = torch.stack(padded_inputs)                      # (B, 3, PADDED, PADDED)
    blurred = F.gaussian_blur(stack, kernel_size=5, sigma=1.0)
    cropped = blurred[:, :, PADDING:-PADDING, PADDING:-PADDING]
    return [cropped[i] for i in range(cropped.shape[0])]

blurred = InfiniteTensor(
    shape=(3, None, None),
    f=blur_batched,
    output_window=TensorWindow((3, WINDOW, WINDOW)),
    args=(base,),
    args_windows=(padded_input_window,),
    tile_store=tile_store,
    tensor_id="blurred_batched",
    batch_size=4,
)
```

Batch size is a per-tensor knob. `base` (with `batch_size=None`) is still called one window at a time.

### 8. `dimension_map` on `TensorWindow`

`TensorWindow.dimension_map` is an optional per-output-dim remap applied by `get_bounds`. `None` in an entry turns that output dim into a singleton (`slice(0, 1)`); an integer entry says which window-space dim feeds that output dim. It is useful when two output dims must step in lockstep, or when an output dim should never slide.

```python
# Singleton leading dim: output window always covers dim 0 as slice(0, 1).
window = TensorWindow(
    size=(1, 64, 64),
    stride=(1, 64, 64),
    dimension_map=(None, 1, 2),
)
```

For most pipelines you can leave `dimension_map=None` (the default); size/stride/offset on the output dim itself already handle fixed-size dims.

### 9. Persistent storage with HDF5

```python
from infinite_tensor import HDF5TileStore, InfiniteTensor, TensorWindow

with HDF5TileStore("tensor_data.h5", tile_size=512) as tile_store:
    tensor = InfiniteTensor(
        shape=(None, None),
        f=your_processing_function,
        output_window=TensorWindow((512, 512)),
        tile_store=tile_store,
        tensor_id="my_infinite_tensor",
    )
    result = tensor[0:2048, 0:2048]
# leaving the `with` block flushes dirty tiles and closes the file
```

`tile_size` controls how HDF5 accumulates overlapping window outputs into on-disk tiles. It may be an int (applied uniformly to every infinite dim) or a tuple sized to the tensor's infinite-dim count. Reopening the same file with the same `tensor_id` must use matching metadata and `tile_size`, or `register_tensor` raises `ValidationError`.

`HDF5TileStore` uses a **write-back** tile cache. Dirty tiles live in memory until:

- they are evicted from the LRU,
- you call `tile_store.flush()` (persist everything, keep the file open),
- or you call `tile_store.close()` / exit the `with` block.

Crashing before any of those loses the dirty tiles. Always wrap long-running persistent workloads in `with HDF5TileStore(...) as store:`, or call `flush()` at checkpoints.

HDF5 tile caching is separate from the `MemoryTileStore` cache:

```python
HDF5TileStore(
    "tensor_data.h5",
    tile_size=512,
    cache_size_bytes=256 * 1024 * 1024,  # 256 MiB tile LRU; default 100 MiB
    cache_size_tiles=None,               # no tile-count limit
)
```

Reconnecting in a later process:

```python
with HDF5TileStore("tensor_data.h5", tile_size=512) as tile_store:
    tensor = InfiniteTensor(
        shape=(None, None),
        f=your_processing_function,      # same f, or a determinism-equivalent one
        output_window=TensorWindow((512, 512)),
        tile_store=tile_store,
        tensor_id="my_infinite_tensor",  # same id as before
    )
    result = tensor[0:2048, 0:2048]      # served from disk; f is only called for new windows
```

Once a window is processed on a persistent store it cannot be un-processed. `clear_cache` only drops in-memory buffers; use `tile_store.clear_tensor(tensor.uuid)` to wipe a tensor's durable state.

### Extending with a custom persistent backend

`PersistentTileStore` implements the tile-accumulation scaffolding (tile math, LRU, window accumulation, write-back) generically. To plug in a different on-disk format, subclass it and implement the seven storage primitives:

```python
from infinite_tensor.tilestore.persistent import PersistentTileStore
```

See `infinite_tensor/tilestore/persistent.py` for the full contract (and `hdf5_tilestore.py` for a worked example).

## Important Notes

1. **Deterministic `f`**: stored outputs assume `f` is pure. Non-determinism silently breaks the "already processed, skip" optimization and can corrupt dependent tensors.
2. **Exact output shape**: `f`'s return must match `TensorWindow.size` exactly (`ShapeMismatchError` otherwise).
3. **Device and dtype**: every `InfiniteTensor` declares a `device` and `dtype` (defaults: `cpu`, `torch.float32`). `f` must return tensors on that exact device and with that exact dtype (`DeviceMismatchError` / `DtypeMismatchError` otherwise). Upstream arg slices are **not** auto-transferred; move or cast them inside `f` if they come from a different device/dtype.
4. **Finite dimensions must fit in memory**: non-`None` dims are materialized whole whenever a window touches them.
5. **Reconnecting**: to reuse stored data, re-construct the tensor with the same `tensor_id` against the same store. The store validates that `to_json()` matches what it saw before and raises `ValidationError` on mismatch.
6. **Avoid manual slicing of dependencies**: use `args` / `args_windows`. Manual indexing inside `f` defeats batching and future memory-management optimizations.
7. **Exploding compute**: `_ensure_processed` recurses into every upstream's window grid with the arg window's pixel bounds. Each chained tensor expands the "cone of interest" by its padding/offset. A deep chain with even modest overlap can detonate the amount of work a single slice triggers. Keep chains shallow and prefer a single fused `f` over many thin layers.
8. **Immutability of persistent stores**: a processed window on `HDF5TileStore` cannot be un-processed without `tile_store.clear_tensor(uuid)`.

## Public exceptions

All live at `infinite_tensor` top level:

| Exception | Raised when |
|-----------|-------------|
| `InfiniteTensorError` | base class for everything below |
| `ValidationError` | constructor param validation, metadata mismatch on re-registration, unsupported dtype migration |
| `ShapeMismatchError` | `f` returned a tensor whose shape differs from `TensorWindow.size` |
| `DeviceMismatchError` | `f` returned a tensor on a different device than the tensor was declared with |
| `DtypeMismatchError` | `f` returned a tensor with a different dtype than the tensor was declared with |
| `TileAccessError` | internal: a tile was requested for read but never processed (indicates a store-consistency bug) |

## Example

See [`examples/blur.py`](examples/blur.py) for a full worked example: a random-image base tensor blurred twice via two chained `InfiniteTensor`s with a padded arg window.

## License

MIT License. See [LICENSE](LICENSE).
