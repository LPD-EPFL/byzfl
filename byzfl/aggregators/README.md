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

To utilize an aggregator or pre-aggregator, first import the desired class from the `aggregators` module and then apply it to your set of input vectors. Here's an example using the Trimmed Mean aggregator:

```python
from byzfl import TrMean
import numpy as np

# Number of Byzantine participants
f = 1

# Input vectors
vectors = np.array([
    [1., 2., 3.],
    [4., 5., 6.],
    [7., 8., 9.]
])

# Initialize the Trimmed Mean aggregator
aggregator = TrMean(f=f)

# Perform aggregation
result = aggregator(vectors)
print("Aggregated result:", result)
```

**Output:**

```
Aggregated result: [4. 5. 6.]
```

## Extending Aggregators and Pre-Aggregators

To add a new aggregator or pre-aggregator, add the desired logic to `aggregators.py` or `pre-aggregators.py`. Ensure your class adheres to the expected interface so it can be seamlessly integrated into the ByzFL framework.

## Documentation

For detailed information on each aggregator and pre-aggregator and their parameters, refer to the [ByzFL documentation](https://byzfl.epfl.ch/).

## License

This module is part of the ByzFL library, licensed under the [MIT License](https://github.com/LPD-EPFL/byzfl/blob/main/LICENSE.txt).