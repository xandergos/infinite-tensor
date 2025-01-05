import abc
from typing import List
from uuid import UUID

class TileStore(abc.ABC):
    """Abstract base class for tile stores."""
    @abc.abstractmethod
    def get(self, key: tuple[int, ...]):
        """Gets a tile from the store.

        Args:
            key (tuple[int, ...]): The key of the tile to get.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def set(self, key: tuple[int, ...], value):
        """Sets a tile in the store.

        Args:
            key (tuple[int, ...]): The key of the tile to set.
            value: The value of the tile to set.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def delete(self, key: tuple[int, ...]):
        """Deletes a tile from the store.

        Args:
            key (tuple[int, ...]): The key of the tile to delete.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def keys(self) -> List[tuple[int, ...]]:
        """Returns a list of all keys in the store."""
        raise NotImplementedError
    
    @abc.abstractmethod
    def is_window_processed(self, window_index: tuple[int, ...]) -> bool:
        raise NotImplementedError
    
    @abc.abstractmethod
    def mark_window_processed(self, window_index: tuple[int, ...]):
        raise NotImplementedError

class MemoryTileStore(TileStore):
    """A tile store that stores tiles in memory."""
    def __init__(self):
        super().__init__()
        self.store = {}
        self.processed_windows = set()
        self.dependent_windows = {}
    
    def get(self, key: tuple[int, ...]):
        return self.store.get(key)

    def set(self, key: tuple[int, ...], value):
        self.store[key] = value
        
    def delete(self, key: tuple[int, ...]):
        if key in self.store:
            del self.store[key]
    
    def keys(self) -> List[tuple[int, ...]]:
        """Returns a list of all keys in the store."""
        return list(self.store.keys())
    
    def is_window_processed(self, window_index: tuple[int, ...]) -> bool:
        return window_index in self.processed_windows
    
    def mark_window_processed(self, window_index: tuple[int, ...]):
        self.processed_windows.add(window_index)
