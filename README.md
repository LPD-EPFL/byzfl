# ByzFL Documentation

Welcome to the **official documentation of ByzFL**, developed by [DCL](http://dcl.epfl.ch) at [EPFL](http://epfl.ch) and [WIDE](https://team.inria.fr/wide/) at [INRIA Rennes](https://www.inria.fr/fr/centre-inria-universite-rennes)!

---

## What is ByzFL?

ByzFL is a **Python library for robust Federated Learning**. It is fully compatible with both [PyTorch](http://pytorch.org) tensors and [NumPy](http://numpy.org) arrays, making it versatile for a wide range of machine learning workflows.

### Key Features

1. **Robust Aggregators and Pre-Aggregators**: 
   - Aggregate gradients robustly while mitigating the impact of Byzantine participants.
2. **Byzantine Attacks**: 
   - Simulate and evaluate different attack strategies to test resilience.
3. **Federated Learning Framework/Simulation**: 
    - Provides an end-to-end simulation environment for federated learning, integrating clients (honest and Byzantine), a central server, and robust aggregation mechanisms.
4. **Federated Learning Benchmarking**:
    - Provides a systematic and automated evaluation framework to test federated learning algorithms under adversarial conditions, ensuring robust performance across various configurations. This framework supports benchmarking the robustness of various aggregation strategies against adversarial attacks in a distributed learning setup.

The exact implementations of these modules (`aggregators`, `attacks`, `benchmark`, and `fed_framework`) can be found in the `byzfl/` directory.

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
# Import necessary libraries
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from byzfl import Client, Server, ByzantineClient, DataDistributor
from byzfl.utils.misc import set_random_seed

# Set random seed for reproducibility
SEED = 42
set_random_seed(SEED)

# Configurations
nb_honest_clients = 3
nb_byz_clients = 1
nb_training_steps = 1000
batch_size = 25

# Data Preparation
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True)

# Distribute data among clients using non-IID Dirichlet distribution
data_distributor = DataDistributor({
    "data_distribution_name": "dirichlet_niid",
    "distribution_parameter": 0.5,
    "nb_honest": nb_honest_clients,
    "data_loader": train_loader,
    "batch_size": batch_size,
})
client_dataloaders = data_distributor.split_data()

# Initialize Honest Clients
honest_clients = [
    Client({
        "model_name": "cnn_mnist",
        "device": "cpu",
        "loss_name": "NLLLoss",
        "LabelFlipping": False,
        "training_dataloader": client_dataloaders[i],
        "momentum": 0.9,
        "nb_labels": 10,
    }) for i in range(nb_honest_clients)
]

# Prepare Test Dataset
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Server Setup, Use SGD Optimizer
server = Server({
    "device": "cpu",
    "model_name": "cnn_mnist",
    "test_loader": test_loader,
    "optimizer_name": "SGD",
    "learning_rate": 0.1,
    "weight_decay": 0.0001,
    "milestones": [1000],
    "learning_rate_decay": 0.25,
    "aggregator_info": {"name": "TrMean", "parameters": {"f": nb_byz_clients}},
    "pre_agg_list": [
        {"name": "Clipping", "parameters": {"c": 2.0}},
        {"name": "NNM", "parameters": {"f": nb_byz_clients}},
        ]
})

# Byzantine Client Setup
attack = {
    "name": "InnerProductManipulation",
    "f": nb_byz_clients,
    "parameters": {"tau": 3.0},
}
byz_client = ByzantineClient(attack)

# Training Loop
for training_step in range(nb_training_steps+1):

    # Send (Updated) Server Model to Clients
    server_model = server.get_dict_parameters()
    for client in honest_clients:
        client.set_model_state(server_model)

    # Evaluate Global Model Every 100 Training Steps
    if training_step % 100 == 0:
        test_acc = server.compute_test_accuracy()
        print(f"--- Training Step {training_step}/{nb_training_steps} ---")
        print(f"Test Accuracy: {test_acc:.4f}")

    # Honest Clients Compute Gradients
    for client in honest_clients:
        client.compute_gradients()

    # Aggregate Honest Gradients
    honest_gradients = [client.get_flat_gradients_with_momentum() for client in honest_clients]

    # Apply Byzantine Attack
    byz_vector = byz_client.apply_attack(honest_gradients)

    # Combine Honest and Byzantine Gradients
    gradients = honest_gradients + byz_vector

    # Update Global Model
    server.update_model(gradients)

print("Training Complete!")
```

## Output
```plaintext
--- Training Step 0/1000 ---
Test Accuracy: 0.0600
--- Training Step 100/1000 ---
Test Accuracy: 0.6375
--- Training Step 200/1000 ---
Test Accuracy: 0.8148
--- Training Step 300/1000 ---
Test Accuracy: 0.9318
--- Training Step 400/1000 ---
Test Accuracy: 0.8588
--- Training Step 500/1000 ---
Test Accuracy: 0.9537
--- Training Step 600/1000 ---
Test Accuracy: 0.9185
--- Training Step 700/1000 ---
Test Accuracy: 0.9511
--- Training Step 800/1000 ---
Test Accuracy: 0.9400
--- Training Step 900/1000 ---
Test Accuracy: 0.9781
--- Training Step 1000/1000 ---
Test Accuracy: 0.9733
Training Complete!
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