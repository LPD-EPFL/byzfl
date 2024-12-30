# Aggregators Module

The `aggregators` module in ByzFL provides a suite of robust aggregation and pre-aggregation methods designed to mitigate the impact of Byzantine participants in federated learning environments.

## Available Aggregators

- **Average**: Computes the standard arithmetic mean of the input vectors.
- **Median**: Calculates the coordinate-wise median, offering resilience against outliers.
- **Trimmed Mean**: Excludes a specified fraction of the highest and lowest values before computing the mean, enhancing robustness.
- **Geometric Median**: Determines the vector minimizing the sum of Euclidean distances to all input vectors, providing strong robustness properties.
- **Krum**: Selects the vector closest to its neighbors, effectively filtering out outliers.
- **Multi-Krum**: An extension of Krum that selects multiple vectors to compute the aggregate for improved robustness.
- **Centered Clipping**: Clips the input vectors to a central value to limit the influence of outliers.
- **MDA (Minimum Diameter Averaging)**: Averages a subset of vectors with smallest diameter for enhanced robustness against Byzantine faults.
- **MONNA**: Averages the closest vectors to a trusted vector for improved resilience.
- **Meamed**: Implements the coordinate-wise mean around median aggregation method for robust federated learning.

## Available Pre-Aggregators

Pre-aggregators perform preliminary computations to refine the input data before the main aggregation step, improving efficiency and robustness:

- **Static Clipping**: Clips input vectors to a predefined norm threshold to limit their influence.
- **Nearest Neighbor Mixing (NNM)**: Pre-processes vectors by averaging with their nearest neighbors to reduce the effect of outliers.
- **Bucketing**: Groups input vectors into buckets, enabling more efficient and robust aggregation.
- **ARC (Adaptive Robust Clipping)**: Dynamically adjusts the clipping threshold based on the data distribution to enhance robustness.

## Usage

To utilize an aggregator or pre-aggregator, first import the desired class from the `byzfl` module and then apply it to your set of input vectors. Here's an example using the Trimmed Mean aggregator and the Nearest Neighbor Mixing (NNM) pre-aggregator:

```python
from byzfl import TrMean, NNM
import numpy as np
import torch

# Number of Byzantine participants
f = 1

# Input vectors - NumPy Array
vectors_np = np.array([
    [1., 2., 3.],
    [4., 5., 6.],
    [7., 8., 9.]
])

# Input vectors - PyTorch Tensor
vectors_torch = torch.tensor([
    [1., 2., 3.],
    [4., 5., 6.],
    [7., 8., 9.]
])

# Initialize the NNM pre-aggregator
pre_aggregator = NNM(f=f)

# Perform pre-aggregation
vectors_np = pre_aggregator(vectors_np)
vectors_torch = pre_aggregator(vectors_torch)
print("Transformed vectors - NumPy Array:", vectors_np)
print("Transformed vectors - PyTorch Tensor:", vectors_torch)

# Initialize the Trimmed Mean aggregator
aggregator = TrMean(f=f)

# Perform aggregation
result_np = aggregator(vectors_np)
result_torch = aggregator(vectors_torch)
print("Aggregated result - NumPy Array:", result_np)
print("Aggregated result - PyTorch Tensor:", result_torch)
```

**Output:**

```
Transformed vectors - NumPy Array: [[2.5 3.5 4.5]
 [2.5 3.5 4.5]
 [5.5 6.5 7.5]]
Transformed vectors - PyTorch Tensor: tensor([[2.5000, 3.5000, 4.5000],
        [2.5000, 3.5000, 4.5000],
        [5.5000, 6.5000, 7.5000]])

Aggregated result - NumPy Array: [2.5 3.5 4.5]
Aggregated result - PyTorch Tensor: tensor([2.5000, 3.5000, 4.5000])
```

## Extending Aggregators and Pre-Aggregators

To add a new aggregator or pre-aggregator, add the desired logic to `aggregators.py` or `pre-aggregators.py`. Ensure your class adheres to the expected interface so it can be seamlessly integrated into the ByzFL framework.

## Documentation

For detailed information on each aggregator and pre-aggregator and their parameters, refer to the [ByzFL documentation](https://byzfl.epfl.ch/).

## License

This module is part of the ByzFL library, licensed under the [MIT License](https://github.com/LPD-EPFL/byzfl/blob/main/LICENSE.txt).