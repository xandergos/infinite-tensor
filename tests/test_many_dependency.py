from infinite_tensors.infinite_tensors import *
import numpy as np
import torch

def test_many_dependency():
    def base_f(ctx):
        return torch.zeros((10, 512, 512))
    
    def dep_f(ctx, base):
        return base + 1
    
    base = InfiniteTensor((10, None, None), base_f, TensorWindow((10, 512, 512)))
    
    dep = base
    for i in range(10):
        dep = InfiniteTensor((10, None, None), dep_f, TensorWindow((10, 512, 512)),
                             args=(dep,), args_windows=(TensorWindow((10, 512, 512)),))
    
    sl = dep[:, 0:512, 0:512]
    print(sl)
    assert torch.all(sl == 10)

if __name__ == "__main__":
    test_many_dependency()