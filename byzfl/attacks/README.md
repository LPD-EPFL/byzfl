# Attacks Module

The `attacks` module in ByzFL provides a comprehensive suite of Byzantine attack strategies designed to test the robustness of federated learning systems. These implementations facilitate the evaluation of aggregation algorithms under adversarial conditions.

## Available Attacks

- **Sign Flipping**: Reverses the sign of gradients to hinder model convergence.
- **Fall of Empires (FOE)**: Generalisation of Sign Flipping, effectively steering the model away from optimal convergence.
- **A Little Is Enough (ALIE)**: Introduces subtle yet effective perturbations to gradients, proportional to the standard deviation of the input vectors.
- **Infinity (Inf)**: Generates vectors with extreme values, effectively disrupting the learning process.
- **Mimic**: Subtle attack strategy where the Byzantine participants aim to mimic the behavior of honest participants, instead of generating obvious outliers.
- **Gaussian**: Generates gradients sampled from a Gaussian distribution, introducing randomness and potential divergence

## Usage

To employ an attack, import the desired class from the `byzfl` module and apply it to your set of input vectors. Here's an example using the Sign Flipping Attack:

```python
from byzfl import SignFlipping
import numpy as np

# Honest vectors
honest_vectors = np.array([
    [1., 2., 3.],
    [4., 5., 6.],
    [7., 8., 9.]
])

# Initialize the Sign Flipping attack
attack = SignFlipping()

# Generate the attack vector
byz_vector = attack(honest_vectors)

print("Byzantine vector:", byz_vector)
```

**Output:**

```
Byzantine vector: [-4. -5. -6.]
```

## Extending Attacks

To introduce a new attack, add the desired logic to `attacks.py`. Ensure your class adheres to the expected interface so it can be seamlessly integrated into the ByzFL framework.

## Documentation

For detailed information on each attack and their parameters, refer to the [ByzFL documentation](https://byzfl.epfl.ch/).

## License

This module is part of the ByzFL library, licensed under the [MIT License](https://github.com/LPD-EPFL/byzfl/blob/main/LICENSE.txt).