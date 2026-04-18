from importlib.metadata import PackageNotFoundError, version

from .infinite_tensor import (
    DeviceMismatchError,
    DtypeMismatchError,
    InfiniteTensor,
    InfiniteTensorError,
    ShapeMismatchError,
    TileAccessError,
    ValidationError,
)
from .tensor_window import TensorWindow
from .tilestore import MemoryTileStore

# Optional HDF5 support
try:
    from .tilestore.hdf5_tilestore import HAS_H5PY, HDF5TileStore  # noqa: F401

    _HAS_HDF5 = HAS_H5PY
except ImportError:
    _HAS_HDF5 = False

try:
    __version__ = version("infinite-tensor")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "InfiniteTensor",
    "InfiniteTensorError",
    "TileAccessError",
    "ShapeMismatchError",
    "DeviceMismatchError",
    "DtypeMismatchError",
    "ValidationError",
    "TensorWindow",
    "MemoryTileStore",
]

if _HAS_HDF5:
    __all__.append("HDF5TileStore")
