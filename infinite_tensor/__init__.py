from .infinite_tensor import InfiniteTensor
from .tensor_window import TensorWindow
from .tilestore import MemoryTileStore
from importlib.metadata import PackageNotFoundError, version

# Optional HDF5 support
try:
    from .tilestore.hdf5_tilestore import HDF5TileStore, HAS_H5PY
    _HAS_HDF5 = HAS_H5PY
except ImportError:
    _HAS_HDF5 = False

try:
    __version__ = version("infinite-tensor")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    'InfiniteTensor',
    'TensorWindow',
    'MemoryTileStore',
]

if _HAS_HDF5:
    __all__.append('HDF5TileStore')
