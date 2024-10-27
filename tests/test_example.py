from infinite_tensors import *
import numpy as np
import torch

def test_infinite_tensor():
    def inverse_gaussian_cdf(x: torch.Tensor) -> torch.Tensor:
        """Compute inverse Gaussian CDF (probit function) for given input tensor.
        
        Args:
            x: Input tensor with values in range [0,1]
            
        Returns:
            Tensor with same shape as input, transformed through inverse normal CDF
        """
        # Convert to numpy for scipy stats function
        x_np = x.detach().numpy()
        
        # Clip input to valid range
        x_np = np.clip(x_np, 0.000001, 0.999999)
        
        # Use scipy's inverse normal CDF (probit function)
        from scipy import stats
        result = stats.norm.ppf(x_np)
        
        # Convert back to tensor
        return torch.from_numpy(result).to(x.dtype)
    
    init_fn = lambda indices, shape: torch.rand(shape)
    t = InfiniteTensor((10, None, None), tile_init_fn=init_fn)
    n = InfiniteTensorResult(inverse_gaussian_cdf, 
                             args=[SlidingWindow(t, 256, 256)],
                             kwargs={},
                             window_size=256,
                             window_stride=128,
                             shape=(10, None, None),
                             output_weights=torch.ones(1, 256, 256))
    sl = n[:, 0:256, 0:256]
    assert sl.shape == (10, 256, 256)
    assert torch.abs(torch.std(sl) - 0.5).item() < 0.1
    print(torch.std(sl), "Expected 0.5")
