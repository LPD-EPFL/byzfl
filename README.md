# ByzFL Documentation

Welcome to the **official documentation of ByzFL**, developed by [DCL](http://dcl.epfl.ch) at [EPFL](http://epfl.ch) and [WIDE](https://team.inria.fr/wide/) at [INRIA Rennes](https://www.inria.fr/fr/centre-inria-universite-rennes)!

---

## What is ByzFL?

ByzFL is a **Python library for Byzantine-resilient Federated Learning**. It is fully compatible with both [PyTorch](http://pytorch.org) tensors and [NumPy](http://numpy.org) arrays, making it versatile for a wide range of machine learning workflows.

### Key Features

1. **Robust Aggregators and Pre-Aggregators**: 
   - Aggregate gradients robustly while mitigating the impact of Byzantine participants.
2. **Byzantine Attacks**: 
   - Simulate and evaluate different attack strategies to test resilience.
3. **Machine Learning Pipelines**: 
   - Train and benchmark robust aggregation schemes and attack implementations seamlessly.

The exact implementations of these modules (`aggregators`, `attacks`, and `pipeline`) can be found in the `byzfl/` directory.

---

## Installation

Install the ByzFL library using pip:

```bash
pip install byzfl
```

After installation, the library is ready to use. Here’s a quick example of how to use the `TrMean` robust aggregator and the `SignFlipping` Byzantine attack:

---

## Quick Start Example - Using PyTorch Tensors

```python
import byzfl
import torch

# Number of Byzantine participants
f = 1

# Honest vectors
honest_vectors = torch.tensor([[1., 2., 3.],
                               [4., 5., 6.],
                               [7., 8., 9.]])

# Initialize and apply the attack
attack = byzfl.SignFlipping()
byz_vector = attack(honest_vectors)

# Create f identical attack vectors
byz_vectors = byz_vector.repeat(f, 1)

# Concatenate honest and Byzantine vectors
all_vectors = torch.cat((honest_vectors, byz_vectors), dim=0)

# Initialize and perform robust aggregation
aggregate = byzfl.TrMean(f=f)
result = aggregate(all_vectors)
print("Aggregated result:", result)
```

**Output:**

```
Aggregated result: tensor([2.5000, 3.5000, 4.5000])
```

---

## Quick Start Example - Using Numpy Arrays

```python
import byzfl
import numpy as np

# Number of Byzantine participants
f = 1

# Honest vectors
honest_vectors = np.array([[1., 2., 3.],
                           [4., 5., 6.],
                           [7., 8., 9.]])

# Initialize and apply the attack
attack = byzfl.SignFlipping()
byz_vector = attack(honest_vectors)

# Create f identical attack vectors
byz_vectors = np.tile(byz_vector, (f, 1))

# Concatenate honest and Byzantine vectors
all_vectors = np.concatenate((honest_vectors, byz_vectors), axis=0)

# Initialize and perform robust aggregation
aggregate = byzfl.TrMean(f=f)
result = aggregate(all_vectors)
print("Aggregated result:", result)
```

**Output:**

```
Aggregated result: [2.5 3.5 4.5]
```

---

## Learn More

Explore the key components of ByzFL:

- [Aggregators](https://byzfl.epfl.ch/aggregators/index.html)  
  Learn about robust aggregation methods.
- [Attacks](https://byzfl.epfl.ch/attacks/index.html)  
  Discover Byzantine attack implementations.
- [Pipeline](https://byzfl.epfl.ch/pipeline/index.html)  
  Build and benchmark your models.

---

## License

ByzFL is open-source and distributed under the [MIT License](https://github.com/LPD-EPFL/byzfl/blob/main/LICENSE.txt).

Contributions are welcome!