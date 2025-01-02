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
3. **Federated Learning Framework**: 
    - Provides an end-to-end simulation environment for federated learning, integrating clients (honest and Byzantine), a central server, and robust aggregation mechanisms. This framework supports testing and benchmarking the robustness of various aggregation strategies against adversarial attacks in a distributed learning setup.

The exact implementations of these modules (`aggregators`, `attacks`, and `fed_framework`) can be found in the `byzfl/` directory.

---

## Installation

Install the ByzFL library using pip:

```bash
pip install byzfl
```

After installation, the library is ready to use.

---


## Federated Learning Simulation Example

Below is an example of how to simulate federated learning using this framework:

```python
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from byzfl import Client, Server, ByzantineClient

# Fix the random seed
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Define data loaders
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# Client parameters
client_params = {
    "model_name": "cnn_mnist",
    "device": "cpu",
    "learning_rate": 0.01,
    "loss_name": "CrossEntropyLoss",
    "weight_decay": 0.0005,
    "milestones": [10, 20],
    "learning_rate_decay": 0.5,
    "LabelFlipping": False,
    "momentum": 0.9,
    "training_dataloader": train_loader,
    "nb_labels": 10,
    "nb_steps": len(train_loader)
}

# Server parameters
server_params = {
    "device": "cpu",
    "model_name": "cnn_mnist",
    "test_loader": test_loader,
    "learning_rate": 0.01,
    "weight_decay": 0.0005,
    "milestones": [10, 20],
    "learning_rate_decay": 0.5,
    "aggregator_info": {"name": "TrMean", "parameters": {"f": 1}},
    "pre_agg_list": [{"name": "Clipping", "parameters": {"c": 2.0}}, {"name": "NNM", "parameters": {"f": 1}}]
}

# Initialize Server and Clients
server = Server(server_params)
clients = [Client(client_params) for _ in range(3)]
byz_worker = ByzantineClient({"name": "InnerProductManipulation", "f": 1, "parameters": {"tau": 3.0}})

# Simulate federated learning
for epoch in range(2):  # Simulate 2 epochs
    # Clients compute their gradients
    for client in clients:
        client.compute_gradients()
    # Collect gradients (with momentum) from honest clients
    honest_gradients = [client.get_flat_gradients_with_momentum() for client in clients]
    # Apply Byzantine attack
    byz_vector = byz_worker.apply_attack(honest_gradients)
    # Combine honest gradients and Byzantine gradients
    gradients = honest_gradients + byz_vector
    # Server aggregates gradients and updates the global model
    server.update_model(gradients)
    # Evaluate global model
    print(f"Epoch {epoch + 1}")
    print("Test Accuracy:", server.compute_test_accuracy())
```

## Output
```plaintext
Epoch 1
Test Accuracy: 0.1015
Epoch 2
Test Accuracy: 0.1015
```

Here are quick examples of how to use the `TrMean` robust aggregator and the `SignFlipping` Byzantine attack:

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
- [Federated Learning Framework](https://byzfl.epfl.ch/fed_framework/index.html)  
  Build and benchmark your models.

---

## License

ByzFL is open-source and distributed under the [MIT License](https://github.com/LPD-EPFL/byzfl/blob/main/LICENSE.txt).

Contributions are welcome!