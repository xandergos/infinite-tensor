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

Overlapping output windows are **summed** in the region they overlap by default. Pass `blend=` / `blend_init=` to `InfiniteTensor` to override — see [Custom window blending](#9-custom-window-blending).

## Key Concepts

1. `**TensorWindow`**: output window specification. Fixed `size`, optional `stride` (defaults to `size`, non-overlapping) and `offset` (defaults to zeros). Your `f` must return a tensor of exactly `size`.
2. `**InfiniteTensor`**: immutable tensor with a shape (any number of `None` dims), a deterministic `f`, a declared `dtype` and `device`, and a backing `TileStore`. Can depend on other `InfiniteTensor`s through `args` / `args_windows`.
3. `**TileStore**`: owns cached results. Pick one:
  - `MemoryTileStore`: in-memory, one shared cache for every tensor using that store.
  - `HDF5TileStore`: persistent, keeps computed results on disk in an HDF5 file so later reads can reuse them. Requires `h5py` (install via the `hdf5` extra).

If you are building your own durable backend, subclass `PersistentTileStore` (`from infinite_tensor.tilestore.persistent import PersistentTileStore`); see [Extending with a custom persistent backend](#extending-with-a-custom-persistent-backend).

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

`tensor_id` defaults to a random id. Pass an explicit id if you want to **reconnect** to stored data on a later run (for persistent stores) or across tensor re-creations in the same session. When you rebuild a tensor with the same id, its configuration must match the previous one.

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

`MemoryTileStore` holds one shared cache for every tensor using that store:

```python
tile_store = MemoryTileStore(
    cache_size_bytes=200 * 1024 * 1024,  # 200 MB
    cache_size_windows=None,             # no window-count limit
)
```

Either (or both) of `cache_size_bytes` / `cache_size_windows` may be `None` (default, unbounded on that axis) or a positive int.

Eviction is paused while a read is in progress, including reads triggered through dependencies. That means a single access can temporarily exceed the cache limit and still finish successfully. Once the read completes, the cache trims back to its limits.

`HDF5TileStore` has its own separate on-disk cache settings; see [Persistent storage with HDF5](#10-persistent-storage-with-hdf5).

### 5. Clearing state

```python
tensor.clear_cache()
tile_store.clear_tensor("my_infinite_tensor")
```

- `tensor.clear_cache()` drops recomputable in-memory state for this tensor. On `MemoryTileStore` that means cached windows are forgotten and will be recomputed on next access. On persistent stores it also flushes pending writes before dropping the in-memory cache, but it does **not** delete data already saved on disk.
- `tile_store.clear_tensor(tensor_id)` deletes **all** state for the tensor, including any persisted data on disk. Use this when you want to fully reset a tensor instead of just clearing RAM caches.

## Advanced Features

### 6. Dependency chaining

Build pipelines by making one infinite tensor depend on another. `f` is called as `f(ctx, *args_sliced)`, where each `args_sliced[i]` is the part of upstream tensor `args[i]` needed for the current output window, as described by `args_windows[i]`.

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

Do **not** manually index `base` inside `f`. Use `args` / `args_windows`; the library handles the upstream slicing for you and keeps dependency reads efficient.

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

`dimension_map` is an advanced option that lets you say which sliding window coordinate controls each output dimension.

- Use an integer to reuse one of the window coordinates for that output dimension.
- Use `None` to keep that output dimension fixed instead of sliding.

This is mainly useful when two output dimensions should move together, or when one dimension should always stay at a single position.

```python
# Fixed leading dim: the first dimension always stays at length 1.
window = TensorWindow(
    size=(1, 64, 64),
    stride=(1, 64, 64),
    dimension_map=(None, 1, 2),
)
```

For most pipelines you can leave `dimension_map=None` (the default). You only need it when the default "one sliding coordinate per output dimension" behavior is not what you want.

### 9. Custom window blending

Overlapping windows are summed by default. Two optional `InfiniteTensor` kwargs let you change that rule:

- `blend: Callable[[Tensor, Tensor], Tensor] | None` — elementwise `(existing, incoming) -> combined`. `None` (default) keeps the usual sum behavior.
- `blend_init: float | int | None` — starting value used before any window contributes to a pixel. `None` (default) means zero. For non-additive blends, set this to the identity of your blend, such as `float("-inf")` for `torch.maximum`.

```python
import torch
from infinite_tensor import InfiniteTensor, TensorWindow, MemoryTileStore

def f(ctx):
    wy, wx = ctx
    return torch.full((64, 64), float(wy + wx))

tensor = InfiniteTensor(
    shape=(None, None),
    f=f,
    output_window=TensorWindow((64, 64), stride=(32, 32)),
    tile_store=MemoryTileStore(),
    tensor_id="max_blend_example",
    blend=torch.maximum,
    blend_init=float("-inf"),
)
result = tensor[0:128, 0:128]  # per-pixel max over every covering window
```

Both kwargs apply to `MemoryTileStore` and any `PersistentTileStore` subclass (including `HDF5TileStore`).

If you persist data to disk, keep the same `blend` / `blend_init` when reopening it. Stored results have already been combined under that rule, so changing it later can produce incorrect answers.

### 10. Persistent storage with HDF5

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
# leaving the `with` block saves pending cached data and closes the file
```

`tile_size` controls how results are grouped on disk. It may be an int (applied uniformly to every infinite dim) or a tuple sized to the tensor's infinite-dim count. Reopening the same file with the same `tensor_id` must use the same tensor configuration and `tile_size`, or construction fails with `ValidationError`.

`HDF5TileStore` keeps recent updates in memory until:

- they are evicted from the cache,
- you call `tile_store.flush()` (persist everything, keep the file open),
- or you call `tile_store.close()` / exit the `with` block.

Crashing before any of those loses the in-memory updates that have not been written yet. Always wrap long-running persistent workloads in `with HDF5TileStore(...) as store:`, or call `flush()` at checkpoints.

The HDF5 store has its own cache limits:

```python
HDF5TileStore(
    "tensor_data.h5",
    tile_size=512,
    cache_size_bytes=256 * 1024 * 1024,  # 256 MiB cache; default 100 MiB
    cache_size_tiles=None,               # no tile-count limit
)
```

Reconnecting in a later process:

```python
with HDF5TileStore("tensor_data.h5", tile_size=512) as tile_store:
    tensor = InfiniteTensor(
        shape=(None, None),
        f=your_processing_function,      # same f, or one that produces the same results
        output_window=TensorWindow((512, 512)),
        tile_store=tile_store,
        tensor_id="my_infinite_tensor",  # same id as before
    )
    result = tensor[0:2048, 0:2048]      # served from disk; f is only called for new windows
```

Once data is saved to a persistent store, `clear_cache` only drops in-memory buffers. Use `tile_store.clear_tensor("my_infinite_tensor")` to fully wipe the saved state for that tensor.

### Extending with a custom persistent backend

This section is only for backend authors.

`PersistentTileStore` provides the shared machinery for disk-backed stores. To plug in a different on-disk format, subclass it and implement the seven required storage methods:

```python
from infinite_tensor.tilestore.persistent import PersistentTileStore
```

See `infinite_tensor/tilestore/persistent.py` for the full contract and `hdf5_tilestore.py` for a worked example.

## Important Notes

1. **Deterministic `f`**: stored outputs assume `f` is pure. Non-determinism can make cached results disagree with fresh recomputation and can corrupt dependent tensors.
2. **Exact output shape**: `f`'s return must match `TensorWindow.size` exactly (`ShapeMismatchError` otherwise).
3. **Device and dtype**: every `InfiniteTensor` declares a `device` and `dtype` (defaults: `cpu`, `torch.float32`). `f` must return tensors on that exact device and with that exact dtype (`DeviceMismatchError` / `DtypeMismatchError` otherwise). Upstream arg slices are **not** auto-transferred; move or cast them inside `f` if they come from a different device/dtype.
4. **Finite dimensions must fit in memory**: non-`None` dims are materialized whole whenever a window touches them.
5. **Reconnecting**: to reuse stored data, re-construct the tensor with the same `tensor_id` against the same store. Its saved configuration must still match, or construction raises `ValidationError`.
6. **Avoid manual slicing of dependencies**: use `args` / `args_windows`. Manual indexing inside `f` bypasses batching and can trigger extra work.
7. **Exploding compute**: each layer in a dependency chain can enlarge the region that has to be read from upstream, especially when windows overlap or include padding. A deep chain can make a small output slice trigger much more work than expected. Keep chains shallow when possible, and prefer one larger `f` over many thin layers if performance becomes a problem.
8. **Immutability of persistent stores**: once data is saved in `HDF5TileStore`, you cannot mark it as missing again without deleting that tensor's stored state with `tile_store.clear_tensor(...)`.

## Public exceptions

All live at `infinite_tensor` top level:


| Exception             | Raised when                                                                                     |
| --------------------- | ----------------------------------------------------------------------------------------------- |
| `InfiniteTensorError` | base class for everything below                                                                 |
| `ValidationError`     | invalid constructor arguments, reconnecting with a different saved configuration, unsupported dtype migration |
| `ShapeMismatchError`  | `f` returned a tensor whose shape differs from `TensorWindow.size`                              |
| `DeviceMismatchError` | `f` returned a tensor on a different device than the tensor was declared with                   |
| `DtypeMismatchError`  | `f` returned a tensor with a different dtype than the tensor was declared with                  |


## Example

See `[examples/blur.py](examples/blur.py)` for a full worked example: a random-image base tensor blurred twice via two chained `InfiniteTensor`s with a padded arg window.

## License

MIT License. See [LICENSE](LICENSE).