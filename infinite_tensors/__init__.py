"""Infinite Tensors: Memory-efficient processing of theoretically infinite tensors.

This library provides a sophisticated system for working with infinite tensors through
a combination of tiling, windowing, and lazy evaluation. It enables processing of
datasets that are too large for memory by computing only the required portions
on-demand.

## Core Concepts

### Infinite Tensors
Tensors with one or more infinite dimensions (specified as None in shape).
Data is generated lazily using user-provided functions.

### Tiling System
Infinite dimensions are divided into finite tiles that fit in memory.
Tiles are loaded/computed on-demand and can be automatically cleaned up.

### Window Processing
Operations are applied through sliding windows that define how to slice
input tensors for processing.

### Dependency Tracking
Automatic memory management through reference counting of tile dependencies.

## Example Usage

```python
from infinite_tensors import InfiniteTensor, TensorWindow
import torch

# Define a function that generates data for each window
def noise_generator(ctx):
    return torch.randn(3, 512, 512)

# Create an infinite tensor
tensor = InfiniteTensor(
    shape=(3, None, None),  # 3 channels, infinite height/width
    f=noise_generator,
    output_window=TensorWindow((3, 512, 512))
)

# Access data - automatically computed on demand
data = tensor[:, 0:1024, 0:1024]  # Gets a 3x1024x1024 region
```

## Memory Management

The library automatically manages memory through:
- Lazy evaluation: Data is only computed when accessed
- Tile-based storage: Large tensors are broken into manageable chunks
- Reference counting: Tiles are freed when no longer needed
- Context managers: Explicit cleanup with `with` statements

## Performance Considerations

- Window size affects memory usage and computation efficiency
- Overlapping windows enable complex operations but increase memory usage
- Dependency chains should be managed to prevent excessive memory usage
- Tiles are CPU-only for broad compatibility

See individual class documentation for detailed usage information.
"""

from .infinite_tensors import InfiniteTensor, InfinityTensorTile
from .tensor_window import TensorWindow
from .tilestore import TileStore, MemoryTileStore
from .utils import normalize_slice, standardize_indices

__all__ = [
    'InfiniteTensor',
    'InfinityTensorTile', 
    'TensorWindow',
    'TileStore',
    'MemoryTileStore',
    'normalize_slice',
    'standardize_indices'
]
