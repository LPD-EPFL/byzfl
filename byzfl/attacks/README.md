# Attacks Module

The `attacks` module in ByzFL provides a comprehensive suite of Byzantine attack strategies designed to test the robustness of federated learning systems. These implementations facilitate the evaluation of aggregation algorithms under adversarial conditions.

## Available Attacks

- **Sign Flipping**: Reverses the sign of gradients to hinder model convergence.
- **Inner Product Manipulation (IPM)**: Alters gradients to manipulate inner products, effectively steering the model away from optimal convergence.
- **Optimal Inner Product Manipulation (Opt-IPM)**: Generalization of the IPM attack, where the attack factor is optimally determined using a line-search optimization method.
- **A Little Is Enough (ALIE)**: Introduces subtle yet effective perturbations to gradients, proportional to the standard deviation of the input vectors.
- **Optimal A Little Is Enough (Opt-ALIE)**: Generalization of the ALIE attack, where the attack factor is optimally determined using a line-search optimization method.
- **Infinity (Inf)**: Generates vectors with extreme values, effectively disrupting the learning process.
- **Mimic**: Subtle attack strategy where the Byzantine participants aim to mimic the behavior of honest participants, instead of generating obvious outliers.
- **Gaussian**: Generates gradients sampled from a Gaussian distribution, introducing randomness and potential divergence

## Usage

To employ an attack, import the desired class from the `byzfl` module and apply it to your set of input vectors. Here's an example using the Sign Flipping Attack:

```python
from byzfl import SignFlipping
import numpy as np
import torch

# Honest vectors - NumPy Array
honest_vectors_np = np.array([
    [1., 2., 3.],
    [4., 5., 6.],
    [7., 8., 9.]
])

# Honest vectors - PyTorch Tensor
honest_vectors_torch = torch.tensor([
    [1., 2., 3.],
    [4., 5., 6.],
    [7., 8., 9.]
])

# Initialize the Sign Flipping attack
attack = SignFlipping()

# Generate the attack vector - NumPy Array
byz_vector_np = attack(honest_vectors_np)

# Generate the attack vector - PyTorch Tensor
byz_vector_torch = attack(honest_vectors_torch)

print("Byzantine vector - NumPy Array:", byz_vector_np)
print("Byzantine vector - PyTorch Tensor:", byz_vector_torch)
```

**Output:**

```
Byzantine vector - NumPy Array: [-4. -5. -6.]
Byzantine vector - PyTorch Tensor: tensor([-4., -5., -6.])
```

## Extending Attacks

To introduce a new attack, add the desired logic to `attacks.py`. Ensure your class adheres to the expected interface so it can be seamlessly integrated into the ByzFL framework.

## Documentation

For detailed information on each attack and their parameters, refer to the [ByzFL documentation](https://byzfl.epfl.ch/).

## License

This module is part of the ByzFL library, licensed under the [MIT License](https://github.com/LPD-EPFL/byzfl/blob/main/LICENSE.txt).