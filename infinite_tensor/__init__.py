from .infinite_tensor import InfiniteTensor
from .tensor_window import TensorWindow
from .tilestore import MemoryTileStore
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("infinite-tensor")
except PackageNotFoundError:
    __version__ = "0.0.0"
    
__all__ = [
    'InfiniteTensor',
    'TensorWindow',
    'MemoryTileStore',
]
