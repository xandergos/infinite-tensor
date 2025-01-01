from infinite_tensors.infinite_tensors import InfinityTensorTile
import abc

class TileStore(abc.ABC):
    @abc.abstractmethod
    def get(self, key: tuple[int, ...]) -> InfinityTensorTile:
        raise NotImplementedError
    
    @abc.abstractmethod
    def set(self, key: tuple[int, ...], value: InfinityTensorTile):
        raise NotImplementedError
    
    @abc.abstractmethod
    def delete(self, key: tuple[int, ...]):
        raise NotImplementedError
    
    @abc.abstractmethod
    def close(self):
        raise NotImplementedError

class MemoryTileStore(TileStore):
    def __init__(self):
        self.store = {}

    def get(self, key: tuple[int, ...]) -> InfinityTensorTile:
        return self.store[key]

    def set(self, key: tuple[int, ...], value: InfinityTensorTile):
        self.store[key] = value
        
    def delete(self, key: tuple[int, ...]):
        del self.store[key]

    def close(self):
        self.store.clear()