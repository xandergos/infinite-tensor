"""Shared test fixtures and configuration for infinite tensor tests."""

import pytest
import torch
import numpy as np
from infinite_tensor.infinite_tensor import InfiniteTensor, TensorWindow
from infinite_tensor.tilestore import MemoryTileStore
import uuid


@pytest.fixture
def torch_random_seed():
    """Ensure reproducible random results in tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    yield
    # No cleanup needed for seeds


@pytest.fixture
def basic_tensor_window():
    """Standard tensor window for testing."""
    return TensorWindow((10, 512, 512))
@pytest.fixture
def tile_store():
    """Shared in-memory tile store for tests."""
    return MemoryTileStore()



@pytest.fixture
def strided_tensor_window():
    """Tensor window with custom stride for testing."""
    return TensorWindow((10, 512, 512), stride=(10, 256, 256))


@pytest.fixture
def base_tensor_func():
    """Basic tensor generation function."""
    def _base_f(ctx):
        return torch.rand((10, 512, 512))
    return _base_f


@pytest.fixture
def zeros_tensor_func():
    """Zero tensor generation function."""
    def _zeros_f(ctx):
        return torch.zeros((10, 512, 512))
    return _zeros_f


@pytest.fixture
def ones_tensor_func():
    """Ones tensor generation function."""
    def _ones_f(ctx):
        return torch.ones((10, 512, 512))
    return _ones_f


@pytest.fixture
def dependency_func():
    """Function for creating dependent tensors."""
    def _dep_f(ctx, base):
        return base * 2 - 1
    return _dep_f


@pytest.fixture
def increment_func():
    """Function for incrementing tensors."""
    def _inc_f(ctx, base):
        return base + 1
    return _inc_f


@pytest.fixture
def basic_infinite_tensor(ones_tensor_func, basic_tensor_window, tile_store):
    """Create a basic infinite tensor for testing."""
    return tile_store.get_or_create(
        uuid.uuid4(),
        (10, None, None),
        ones_tensor_func,
        basic_tensor_window,
    )


@pytest.fixture
def random_infinite_tensor(base_tensor_func, basic_tensor_window, torch_random_seed, tile_store):
    """Create a random infinite tensor for testing."""
    return tile_store.get_or_create(
        uuid.uuid4(),
        (10, None, None),
        base_tensor_func,
        basic_tensor_window,
    )


@pytest.fixture
def zeros_infinite_tensor(zeros_tensor_func, basic_tensor_window, tile_store):
    """Create a zeros infinite tensor for testing."""
    return tile_store.get_or_create(
        uuid.uuid4(),
        (10, None, None),
        zeros_tensor_func,
        basic_tensor_window,
    )
