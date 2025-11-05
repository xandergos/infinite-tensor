# Infinite Tensors - Maintainability Guide

## High-level idea

An "infinite" tensor is a lazy, windowed view over data with one or more unbounded dimensions. You provide a function `f(ctx, *args, **kwargs)` that produces values for a fixed window shape. Reads slice the requested region into overlapping windows, call `f` only for the needed windows, accumulate results into tiles (chunks), and return the assembled slice. Tiles are cached in the backing `TileStore` until the store or tensor is discarded.

## Modules and responsibilities

### `infinite_tensor/infinite_tensor.py`

- **InfiniteTensor**: main user-facing class.
  - Shape with `None` for infinite dims, dtype, per-infinite-dim chunk_size.
  - `output_window`: TensorWindow controls window size/stride/offset/mapping for the function output.
  - Dependency system: pass other InfiniteTensors as args/kwargs with matching TensorWindows (`args_windows`, `kwargs_windows`) to slice inputs for `f`.
  - Indexing (`__getitem__`): standardizes indices, determines needed tiles and windows, computes missing windows via `_apply_f`, assembles from tiles.
  - Writing (`__setitem__`): allowed only if there are no dependencies; validates shapes and writes into tiles.
  - `_add_op`: internal, creates/accumulates into tiles.
  - Memory management: tiles stored in a TileStore keyed by `(tensor_uuid, tile_index)`.
- **InfinityTensorTile**: stores tile values: `torch.Tensor`.
- Validation helpers and custom exceptions.

### `infinite_tensor/tensor_window.py`

- **TensorWindow**: sliding-window spec.
  - `size`, `stride` (default: size), `offset`, optional `dimension_map`.
  - Window math: `get_lowest_intersection`, `get_highest_intersection`, `pixel_range_to_window_range`, `get_bounds`, and `map_window_slices`.
  - Used for: determining which windows intersect a pixel region and converting window indices to pixel bounds; mapping between dependent tensors.

### `infinite_tensor/tilestore.py`

- **TileStore** (ABC): interface for tile backends: `get`/`set`/`delete`/`keys`, processed-window tracking.
- **MemoryTileStore**: in-RAM dict/set implementation (default). Tracks seen windows to avoid recomputation.

### `infinite_tensor/utils.py`

- Indexing utilities: `normalize_slice`, `standardize_indices` for consistent slice handling (ellipsis, negatives, steps, ints→slices) and tracking collapse (squeeze) dims.

### `infinite_tensor/__init__.py`

- Public API re-exports: `InfiniteTensor`, `TensorWindow`, `TileStore`, `MemoryTileStore`, `normalize_slice`, `standardize_indices`.

## Data flow (reading a slice)

1. Standardize indices via `standardize_indices`.
2. Compute needed tiles from pixel ranges and chunk sizes.
3. Determine window index range intersecting the pixel region using `TensorWindow.get_lowest_intersection`/`get_highest_intersection`.
4. For each required window, if unseen:
   - Slice dependent tensors using their provided `TensorWindow.get_bounds(window_index)`.
   - Call `f(window_index, *sliced_args, **sliced_kwargs)`.
   - Validate output shape matches `output_window.size`.
   - Accumulate into tiles via `_add_op`.
5. Assemble the output tensor by intersecting requested indices with tile bounds and copying from tile-local regions; squeeze collapsed dims.

## Data flow (writing)

- `__setitem__`: only when no dependencies. Validates shape/device, intersects with tiles, writes into existing tiles, updates store.

## Dependencies and cleanup

- Each dependent tensor registers its `TensorWindow` on its inputs. This allows upstream tensors to slice themselves correctly when dependents request data.

## Public API surface

- Construct with `InfiniteTensor(shape, f, output_window, args=..., kwargs=..., args_windows=..., kwargs_windows=..., chunk_size=..., dtype=..., tile_store=...)`.
- Slice like a normal tensor: `tensor[:, y0:y1, x0:x1]`.
- Compose with dependencies by passing other InfiniteTensors and corresponding TensorWindows.
- Context management simply mirrors standard Python patterns; no automatic cleanup is triggered on exit.

## Notable constraints/quirks

- Tiles and function outputs are CPU tensors; device mismatch raises an error.
- Function outputs must exactly match `output_window.size`.
- Chunking applies only to infinite dims; finite dims are fully contained in each tile.

## Testing and examples

- Tests in `tests/` exercise slicing, dependencies, and strides, and demonstrate typical usage with `TensorWindow((C, H, W))` and dependency chains.
- Example in `README.md` shows multi-stage, seamless blur using padded input windows and window offsets to handle boundaries.
- Default backend is `MemoryTileStore`; alternative stores (disk, distributed) can be implemented by subclassing `TileStore`.

## File layout

```
infinite_tensor/: core library (infinite_tensor.py, tensor_window.py, tilestore.py, utils.py, __init__.py)
tests/: behavior/specs and fixtures
examples/: sample usage
```t.