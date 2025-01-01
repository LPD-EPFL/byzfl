# Federated Learning Framework (fed_framework)

The **Federated Learning Framework (fed_framework)** provides a comprehensive suite of tools and components for simulating federated learning workflows. This framework enables researchers and practitioners to test, evaluate, and explore federated learning techniques in the presence of adversarial attacks and aggregation strategies.

## Overview

This framework integrates several key components, including:

1. **Clients and Byzantine Clients**: Simulates the behavior of honest and Byzantine nodes in federated learning.
2. **Server**: Central component for aggregating updates and managing the global model.
3. **Robust Aggregators**: Implements aggregation techniques to mitigate the effects of adversarial updates.
4. **Models**: Includes a variety of neural network architectures tailored for different datasets.
5. **Workflow Simulation**: Facilitates the end-to-end simulation of federated learning with built-in tools for dataset handling, aggregation, and attack scenarios.

## Features

- **Federated Learning Workflow**:
  - Simulates distributed learning with multiple clients and a central server.
  - Supports varying levels of dataset heterogeneity across clients.

- **Attack and Defense**:
  - Simulates Byzantine attacks to test the robustness of federated learning.
  - Includes robust aggregation strategies to counter adversarial updates.

- **Model Flexibility**:
  - Predefined architectures for datasets such as MNIST and CIFAR.
  - Supports logistic regression, convolutional networks, and ResNet models.

## Installation

To use the `fed_framework`, clone the repository and ensure all required dependencies are installed. You can install the framework as follows:

```bash
git clone https://github.com/LPD-EPFL/fed_framework.git
cd fed_framework
```

## Components

### Client
The `Client` class simulates an honest federated learning node. Each client trains its local model and shares updates with the server.
*Code for the `Client` class is located in `framework.py`.*

### ByzantineClient
The `ByzantineClient` simulates malicious nodes that perform adversarial attacks on the federated learning process.
*Code for the `ByzantineClient` class is located in `framework.py`.*

### Server
The `Server` class aggregates client updates and maintains the global model, applying robust aggregation techniques to mitigate the effects of Byzantine clients.
*Code for the `Server` class is located in `framework.py`.*

### RobustAggregator
Robust aggregators implement defensive techniques against adversarial updates. Examples include:
- **Trimmed Mean (TrMean)**
- **Clipping (Clipping)**
- **Nearest Neighbor Mixing (NNM)**

*Code for the `RobustAggregator` class is located in `framework.py`.*

### Models
A variety of models are provided, including:

- **`fc_mnist`**, **`cnn_mnist`**, and **`logreg_mnist`** for MNIST.
- **`cnn_cifar`** and ResNet variants (**`ResNet18`**, **`ResNet34`**, etc.) for CIFAR datasets.

Refer to the models documentation for details. *Code for the Models module is located in `models.py`.*

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/LPD-EPFL/byzfl/blob/main/LICENSE.txt) file for details.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request for any feature additions or bug fixes.