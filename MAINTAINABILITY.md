# Infinite Tensors - Maintainability Guide

## High-level idea

An "infinite" tensor is a lazy, windowed view over data with one or more unbounded dimensions. You provide a function `f(ctx, *args, **kwargs)` that produces values for a fixed window shape. Reads slice the requested region into overlapping windows, call `f` only for the needed windows, accumulate results into tiles (chunks), and return the assembled slice. Tiles are reference-tracked and can be cleaned up automatically when no longer needed.

## Modules and responsibilities

### `infinite_tensors/infinite_tensors.py`

- **InfiniteTensor**: main user-facing class.
  - Shape with `None` for infinite dims, dtype, per-infinite-dim chunk_size.
  - `output_window`: TensorWindow controls window size/stride/offset/mapping for the function output.
  - Dependency system: pass other InfiniteTensors as args/kwargs with matching TensorWindows (`args_windows`, `kwargs_windows`) to slice inputs for `f`.
  - Indexing (`__getitem__`): standardizes indices, determines needed tiles and windows, computes missing windows via `_apply_f`, assembles from tiles.
  - Writing (`__setitem__`): allowed only if there are no dependencies/cleanup marks; validates shapes and writes into tiles.
  - `_add_op`: internal, creates/accumulates into tiles.
  - Memory management: tiles stored in a TileStore keyed by `(tensor_uuid, tile_index)`. 
  - Reference counting via `InfinityTensorTile.dependency_windows_processed` and per-dependency window accounting. 
  - `mark_for_cleanup()` and `_full_cleanup()` delete tiles and recursively release dependencies when safe.
- **InfinityTensorTile**: stores tile values: `torch.Tensor` and a processed counter.
- Validation helpers and custom exceptions.

### `infinite_tensors/tensor_window.py`

- **TensorWindow**: sliding-window spec.
  - `window_size`, `window_stride` (default: size), `window_offset`, optional `dimension_map`.
  - Window math: `get_lowest_intersection`, `get_highest_intersection`, `pixel_range_to_window_range`, `get_bounds`, and `map_window_slices`.
  - Used for: determining which windows intersect a pixel region and converting window indices to pixel bounds; mapping between dependent tensors.

### `infinite_tensors/tilestore.py`

- **TileStore** (ABC): interface for tile backends: `get`/`set`/`delete`/`keys`, processed-window tracking.
- **MemoryTileStore**: in-RAM dict/set implementation (default). Tracks seen windows to avoid recomputation.

### `infinite_tensors/utils.py`

- Indexing utilities: `normalize_slice`, `standardize_indices` for consistent slice handling (ellipsis, negatives, steps, ints→slices) and tracking collapse (squeeze) dims.

### `infinite_tensors/__init__.py`

- Public API re-exports: `InfiniteTensor`, `TensorWindow`, `TileStore`, `MemoryTileStore`, `normalize_slice`, `standardize_indices`.

## Data flow (reading a slice)

1. Standardize indices via `standardize_indices`.
2. Compute needed tiles from pixel ranges and chunk sizes.
3. Determine window index range intersecting the pixel region using `TensorWindow.get_lowest_intersection`/`get_highest_intersection`.
4. For each required window, if unseen:
   - Slice dependent tensors using their provided `TensorWindow.get_bounds(window_index)`.
   - Call `f(window_index, *sliced_args, **sliced_kwargs)`.
   - Validate output shape matches `output_window.window_size`.
   - Accumulate into tiles via `_add_op`.
   - Mark dependencies' tiles as "processed" for cleanup accounting.
5. Assemble the output tensor by intersecting requested indices with tile bounds and copying from tile-local regions; squeeze collapsed dims.

## Data flow (writing)

- `__setitem__`: only when no dependencies and not cleanup-marked. Validates shape/device, intersects with tiles, writes into existing tiles, updates store.

## Dependencies and cleanup

- Each dependent tensor registers its TensorWindow on its inputs. Inputs increment `dependency_windows_processed` per affected tile when dependents process windows.
- `_is_tile_needed`: computes how many dependent windows overlap a tile (including infinite-dependent-dim cases) and compares to processed count.
- `mark_for_cleanup()`: immediately deletes tiles that are no longer needed; `_full_cleanup()` clears the store and trims dependency registrations when safe.

## Public API surface

- Construct with `InfiniteTensor(shape, f, output_window, args=..., kwargs=..., args_windows=..., kwargs_windows=..., chunk_size=..., dtype=..., tile_store=...)`.
- Slice like a normal tensor: `tensor[:, y0:y1, x0:x1]`.
- Compose with dependencies by passing other InfiniteTensors and corresponding TensorWindows.
- Optional context management: `with InfiniteTensor(...) as t: ...` ensures cleanup.

## Notable constraints/quirks

- Tiles and function outputs are CPU tensors; device mismatch raises an error.
- Function outputs must exactly match `output_window.window_size`.
- Chunking applies only to infinite dims; finite dims are fully contained in each tile.

## Testing and examples

- Tests in `tests/` exercise slicing, dependencies, strides, and cleanup, and demonstrate typical usage with `TensorWindow((C, H, W))` and dependency chains.
- Example in `README.md` shows multi-stage, seamless blur using padded input windows and window offsets to handle boundaries.
- Default backend is `MemoryTileStore`; alternative stores (disk, distributed) can be implemented by subclassing `TileStore`.

## File layout

```
infinite_tensors/: core library (infinite_tensors.py, tensor_window.py, tilestore.py, utils.py, __init__.py)
tests/: behavior/specs and fixtures
examples/: sample usage
```t.