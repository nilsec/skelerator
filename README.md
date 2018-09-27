# Skelerator
Neural toy data generator.

## Installation 

1. Install graph-tool.
    

2. Install the package.
    
    ```    
    git clone https://github.com/nilsec/skelerator.git

    cd skelerator

    python setup.py install
    ```


## Example Usage

Generate a toy segmentation with skeletons via:

```python
    from skelerator import create_segmentation

    shape = np.array([100, 100, 100])
    n_objects = 20
    points_per_skeleton = 5
    interpolation = "linear"
    smoothness = 2
    write_to = "./toy_segmentation"

    create_segmentation(shape, n_objects, points_per_skeleton, interpolation, smoothness, write_to)
```
