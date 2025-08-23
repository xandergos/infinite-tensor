# Infinite Tensors

A Python library for performing operations on theoretically infinite tensors using a sliding window approach. This library enables processing of large tensors without loading the entire tensor into memory.

## Installation

Install using pip:
```bash
pip install git+https://github.com/xandergos/infinite-tensor.git
```


## What is an Infinite Tensor?

An Infinite Tensor is a powerful tool that lets you work with data that has one or more unbounded (infinite) dimensions. Instead of loading all data into memory at once, it:
- Loads only the parts you need, when you need them
- Processes data in manageable chunks (windows)
- Automatically manages memory by cleaning up unused data

Think of it like a smart window that slides over your data, processing only what's visible through that window at any time.

## Key Concepts

### Windows and Chunks

1. **Windows**: Define how your processing function sees the data
   - Fixed size (e.g., 64x64 pixels for image processing)
   - Can overlap if needed
   - Your function processes one window at a time

2. **Chunks**: How data is stored internally
   - Larger blocks that contain processed results
   - Automatically managed for memory efficiency
   - You don't need to interact with these directly

## Getting Started

### 1. Creating an Infinite Tensor

```python
from infinite_tensors import InfiniteTensor, TensorWindow

# Basic structure:
tensor = InfiniteTensor(
    shape=(None, None),  # None for infinite dimensions
    f=your_processing_function,  # Function that processes each window
    output_window=TensorWindow((height, width)),  # Window size
    chunk_size=(1000, 1000)  # Size of internal storage chunks
)
```

### 2. Defining Your Processing Function

```python
def your_processing_function(window_index, *args, **kwargs):
    # Process data for the current window
    # Must return a tensor matching output_window size
    return result
```

### 3. Using the Tensor

```python
# Slice it like a normal tensor
result = tensor[y0:y1, x0:x1]

# Use it in a context manager for automatic cleanup
with InfiniteTensor(...) as t:
    result = t[10:100, 20:200]
```

## Advanced Features

### 1. Dependency Chaining

You can create processing pipelines by making one infinite tensor depend on another:

```python
# First processing stage
stage1 = InfiniteTensor(
    shape=(None, None),
    f=process_stage1,
    output_window=TensorWindow((64, 64))
)

# Second stage depends on first
stage2 = InfiniteTensor(
    shape=(None, None),
    f=process_stage2,
    output_window=TensorWindow((32, 32)),
    args=(stage1,),  # Pass stage1 as input
    args_windows=(TensorWindow((64, 64)),)  # Define how stage2 sees stage1
)
```

### 2. Memory Management

- Memory is automatically managed
- Processed data is cached until no longer needed
- Use context managers (`with` statement) for automatic cleanup
- Call `mark_for_cleanup()` to manually free memory

## Important Notes

1. **CPU Only**: All processing happens on CPU. GPU tensors will raise errors.
2. **Window Size**: Your function must return exactly the size specified in `output_window`.
3. **Finite Dimensions**: Non-infinite dimensions must fit in memory.

## Common Patterns

1. **Image Processing**:
   ```python
   # Process large images in windows
   image_processor = InfiniteTensor(
       shape=(None, None, 3),  # Height, Width, RGB
       f=process_image,
       output_window=TensorWindow((64, 64, 3))
   )
   ```

2. **Data Streaming**:
   ```python
   # Process continuous data streams
   stream_processor = InfiniteTensor(
       shape=(None, feature_size),
       f=process_stream,
       output_window=TensorWindow((1000, feature_size))
   )
   ```

## Troubleshooting

Common issues and solutions:

1. **Memory Issues**:
   - Reduce window size
   - Reduce chunk size
   - Use context managers or .mark_for_cleanup() for automatic cleanup

2. **Shape Mismatches**:
   - Ensure your function returns exactly the window size
   - Check that window sizes match between dependent tensors

3. **Performance**:
   - Adjust chunk size to balance memory use and processing overhead
   - Consider window overlap requirements carefully

## Examples

Check out `examples/blur.py` for a complete example showing how to:
- Process images larger than memory
- Handle boundaries correctly
- Chain multiple processing steps

## License

MIT License - See LICENSE file for details.