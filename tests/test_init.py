from infinite_tensors.infinite_tensors import *
import numpy as np
import torch

def test_init():
    def f(ctx):
        return torch.ones((10, 512, 512))
    
    t = InfiniteTensor((10, None, None), f, TensorWindow((10, 512, 512)))
    assert tuple(t[0, 0:512, 0:512].shape) == (512, 512)
    assert tuple(t[:, -1024:1024, -1024:1024].shape) == (10, 2048, 2048)
    assert tuple(t[:5, -512:512, -512:512].shape) == (5, 1024, 1024)
    
    assert torch.all(t[0, 0:512, 0:512] == torch.ones(512, 512))

if __name__ == "__main__":
    test_init()