# Federated Learning Framework (fed_framework)

The **Federated Learning Framework (fed_framework)** provides a comprehensive suite of tools and components for simulating federated learning workflows. This framework enables researchers and practitioners to test, evaluate, and explore federated learning techniques in the presence of adversarial attacks and aggregation strategies.

## Overview

This framework integrates several key components, including:

1. **Clients and Byzantine Clients**: Simulates the behavior of honest and Byzantine nodes in federated learning.
2. **Server**: Central component for aggregating updates and managing the global model.
3. **Robust Aggregators**: Implements aggregation techniques to mitigate the effects of adversarial updates.
4. **Data Distributor:** Provides flexible dataset partitioning strategies for simulating heterogeneous client data distributions.
5. **Models**: Includes a variety of neural network architectures tailored for different datasets.
6. **Workflow Simulation**: Facilitates the end-to-end simulation of federated learning with built-in tools for dataset handling, aggregation, and attack scenarios.

## Components

### Client
The `Client` class simulates an honest federated learning node. Each client trains its local model and shares updates with the server.
*Code for the `Client` class is located in `client.py`.*

### ByzantineClient
The `ByzantineClient` simulates malicious nodes that perform adversarial attacks on the federated learning process.
*Code for the `ByzantineClient` class is located in `byzantine_client.py`.*

### Server
The `Server` class aggregates client updates and maintains the global model, applying robust aggregation techniques to mitigate the effects of Byzantine clients.
*Code for the `Server` class is located in `server..py`.*

### RobustAggregator
The `RobustAggregator` class implements defensive techniques against adversarial updates. Examples include:
- **Trimmed Mean (TrMean)**
- **Clipping (Clipping)**
- **Nearest Neighbor Mixing (NNM)**

*Code for the `RobustAggregator` class is located in `robust_aggregator.py`.*

### DataDistributor
The `DataDistributor` class provides tools for partitioning datasets among clients. It supports various strategies to simulate realistic federated learning scenarios: 
- **IID (Independent and Identically Distributed):** Equally distributes data randomly among clients.
- **Dirichlet Non-IID:** Partitions data using a Dirichlet distribution to introduce heterogeneity.
- **Gamma-Similarity Non-IID:** Balances data similarity and heterogeneity with a tunable gamma parameter.
- **Extreme Non-IID:** Creates highly skewed data distributions by assigning sorted partitions to clients.

*Code for the `DataDistributor` class is located in `data_distributor.py`.*

### Models
A variety of models are provided, including:

- **`fc_mnist`**, **`cnn_mnist`**, and **`logreg_mnist`** for MNIST.
- **`cnn_cifar`** and ResNet variants (**`ResNet18`**, **`ResNet34`**, etc.) for CIFAR datasets.

Refer to the models documentation for details. *Code for the Models module is located in `models.py`.*

## Federated Learning Simulation

> **⚡ Highlight**:  
> An example of a federated learning simulation using these presented components can be found in the [README of the main GitHub repository](https://github.com/LPD-EPFL/byzfl/tree/main).

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/LPD-EPFL/byzfl/blob/main/LICENSE.txt) file for details.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request for any feature additions or bug fixes.