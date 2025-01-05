from infinite_tensors.infinite_tensors import *
import numpy as np
import torch

def test_dependency():
    def base_f(ctx):
        return torch.rand((10, 512, 512))
    
    def dep_f(ctx, base):
        return base * 2 - 1
    
    base = InfiniteTensor((10, None, None), base_f, TensorWindow((10, 512, 512)))
    dep = InfiniteTensor((10, None, None), dep_f, TensorWindow((10, 512, 512)),
                         args=(base,), args_windows=(TensorWindow((10, 512, 512)),))
    
    sl = dep[:, 0:512, 0:512]
    assert sl.max() > 0.99
    assert sl.min() < -0.99
    assert abs(sl.mean()) < 0.01

if __name__ == "__main__":
    test_dependency()