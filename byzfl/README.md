# ByzFL Module

The `byzfl` module is the core component of the ByzFL library, providing functionalities for Byzantine-resilient federated learning.

## Directory Structure

- **aggregators/**: Contains implementations of robust aggregation methods to mitigate the impact of Byzantine participants.
- **attacks/**: Includes various Byzantine attack strategies for testing and evaluation.
- **pipeline/**: Provides tools and scripts for training and benchmarking aggregators and attacks.
- **utils/**: Utility functions and helpers used across the module.
- **__init__.py**: Initializes the `byzfl` module, making its components accessible when imported.

## Installation

Ensure that the ByzFL library is installed in your environment:

```bash
pip install byzfl
```

## Usage

Here's an example of how to use the `byzfl` module to perform a robust aggregation using the `Trmean` aggregator, when the `SignFlipping` attack is executed by the Byzantine participants.

```python
import byzfl
import numpy as np

# Number of Byzantine participants
f = 1

# Honest vectors
honest_vectors = np.array([
    [1., 2., 3.],
    [4., 5., 6.],
    [7., 8., 9.]
])

# Initialize and apply a Byzantine attack (e.g., Sign Flipping)
attack = byzfl.SignFlipping()
byz_vector = attack(honest_vectors)

# Create f identical attack vectors
byz_vectors = np.tile(byz_vector, (f, 1))

# Concatenate honest and Byzantine vectors
all_vectors = np.concatenate((honest_vectors, byz_vectors), axis=0)

# Initialize and perform robust aggregation using Trimmed Mean
aggregator = byzfl.TrMean(f=f)
result = aggregator(all_vectors)
print("Aggregated result:", result)
```

**Output:**

```
Aggregated result: [2.5 3.5 4.5]
```

## Documentation

For detailed information on each component, refer to the [ByzFL documentation](https://byzfl.epfl.ch/).

## License

This module is part of the ByzFL library, licensed under the [MIT License](https://github.com/LPD-EPFL/byzfl/blob/main/LICENSE.txt).