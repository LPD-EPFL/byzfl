# ByzFL Module

The `byzfl` module is the core component of the ByzFL library, providing functionalities for Byzantine-resilient federated learning.

## Directory Structure

- **aggregators/**: Contains implementations of robust aggregation methods to mitigate the impact of Byzantine participants.
- **attacks/**: Includes various Byzantine attack strategies for testing and evaluation.
- **benchmark/**: Contains the benchmarking framework for evaluating federated learning aggregation methods under adversarial conditions, allowing automated large-scale experiments.
- **fed_framework/**: Provides tools and scripts for testing robust (pre-)aggregators and attacks in simulated federated learning environments.
- **utils/**: Utility functions and helpers used across the module.
- **__init__.py**: Initializes the `byzfl` module, making its components accessible when imported.

## Installation

Ensure that the ByzFL library is installed in your environment:

```bash
pip install byzfl
```

## Usage

Here's an example of how to use the `byzfl` module to perform a robust aggregation using the `TrMean` aggregator, when the `SignFlipping` attack is executed by the Byzantine participants.

### Using PyTorch Tensors

```python
import byzfl
import torch

# Number of Byzantine participants
f = 1

# Honest vectors
honest_vectors = torch.tensor([[1., 2., 3.],
                               [4., 5., 6.],
                               [7., 8., 9.]])

# Initialize and apply a Byzantine attack (e.g., Sign Flipping)
attack = byzfl.SignFlipping()
byz_vector = attack(honest_vectors)

# Create f identical attack vectors
byz_vectors = byz_vector.repeat(f, 1)

# Concatenate honest and Byzantine vectors
all_vectors = torch.cat((honest_vectors, byz_vectors), dim=0)

# Initialize and perform robust aggregation using Trimmed Mean
aggregate = byzfl.TrMean(f=f)
result = aggregate(all_vectors)
print("Aggregated result:", result)
```

**Output:**

```
Aggregated result: tensor([2.5000, 3.5000, 4.5000])
```

### Using NumPy Arrays

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

## Citation

If you use **ByzFL** in your research, please cite:

```bibtex
@misc{gonzález2025byzflresearchframeworkrobust,
  title     = {ByzFL: Research Framework for Robust Federated Learning},
  author    = {Marc González and Rachid Guerraoui and Rafael Pinot and Geovani Rizk and John Stephan and François Taïani},
  year      = {2025},
  eprint    = {2505.24802},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url       = {https://arxiv.org/abs/2505.24802}
}
```

## License

This module is part of the ByzFL library, licensed under the [MIT License](https://github.com/LPD-EPFL/byzfl/blob/main/LICENSE.txt).