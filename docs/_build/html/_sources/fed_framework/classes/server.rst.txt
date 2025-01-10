.. _server-label:

Server
======

The ``Server`` class simulates the central server in a federated learning environment. It aggregates client updates, applies robust aggregation techniques, and updates the global model to ensure robustness against Byzantine attacks.

Key Features
------------
- **Global Model Management**:  
  Updates and maintains the global model using aggregated gradients received from clients.
- **Robust Aggregation**:  
  Integrates defensive techniques, such as :ref:`trmean-label` and :ref:`clipping-label`, to ensure resilience against malicious updates.
- **Performance Evaluation**:  
  Computes accuracy metrics on validation and test datasets to track the global model's progress.
- **Integration**:
  Works in conjunction with ``Client`` and ``ByzantineClient`` classes to simulate realistic federated learning scenarios.

Initialization Parameters
-------------------------
:param params: dict
    A dictionary containing the configuration for the Server. Must include:

- ``"model_name"`` : str  
    Name of the model to be used. Refer to :ref:`models-label` for available models.
- ``"device"`` : str  
    Name of the device to be used for computations (e.g., ``"cpu"``, ``"cuda"``).
- ``"optimizer_name"`` : str  
    Name of the optimizer to use (e.g., ``"SGD"``, ``"Adam"``).
- ``"optimizer_params"`` : dict, optional  
    Parameters for the optimizer (e.g., ``betas`` for Adam, ``momentum`` for SGD).
- ``"learning_rate"`` : float  
    Learning rate for the global model optimizer.
- ``"weight_decay"`` : float  
    Weight decay (L2 regularization) for the optimizer.
- ``"milestones"`` : list  
    List of training steps at which the learning rate decay is applied.
- ``"learning_rate_decay"`` : float  
    Factor by which the learning rate is reduced at each milestone.
- ``"aggregator_info"`` : dict  
    Dictionary specifying the aggregation method and its parameters:
      - ``"name"`` : str  
          Name of the aggregator (e.g., ``"TrMean"``).
      - ``"parameters"`` : dict  
          Parameters for the aggregator.
- ``"pre_agg_list"`` : list  
    List of dictionaries specifying pre-aggregation methods and their parameters:
      - ``"name"`` : str  
          Name of the pre-aggregator (e.g., ``"Clipping"``).
      - ``"parameters"`` : dict  
          Parameters for the pre-aggregator.
- ``"test_loader"`` : DataLoader  
    DataLoader for the test dataset to evaluate the global model.
- ``"validation_loader"`` : DataLoader, optional  
    DataLoader for the validation dataset to monitor training performance.

Methods
-------
- ``aggregate(vectors)``  
  Aggregates input vectors using the configured robust aggregator.

- ``update_model(gradients)``  
  Updates the global model by aggregating gradients and performing an optimization step.

- ``compute_validation_accuracy()``  
  Computes accuracy on the validation dataset.

- ``compute_test_accuracy()``  
  Computes accuracy on the test dataset.

Examples
--------

.. code-block:: python

    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from byzfl import Client, Server, ByzantineClient

    # Define data loader using MNIST
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Define pre-aggregators and aggregator
    pre_aggregators = [{"name": "Clipping", "parameters": {"c": 2.0}}, {"name": "NNM", "parameters": {"f": 1}}]
    aggregator_info = {"name": "TrMean", "parameters": {"f": 1}}

    # Define server parameters
    server_params = {
        "device": "cpu",
        "model_name": "cnn_mnist",
        "test_loader": test_loader,
        "optimizer_name": "Adam",
        "optimizer_params": {"betas": (0.9, 0.999)},
        "learning_rate": 0.01,
        "weight_decay": 0.0005,
        "milestones": [10, 20],
        "learning_rate_decay": 0.5,
        "aggregator_info": aggregator_info,
        "pre_agg_list": pre_aggregators,
    }

    # Initialize the Server
    server = Server(server_params)

    # Example: Aggregation and model update
    gradients = [...]  # Collect gradients from clients
    server.update_model(gradients)
    print("Test Accuracy:", server.compute_test_accuracy())

Notes
-----
- The server is designed to be resilient against Byzantine behaviors by integrating pre-aggregation and aggregation techniques.
- Accuracy evaluation is built-in to monitor the model's progress throughout the simulation.

.. autoclass:: byzfl.Server
   :members:
   :undoc-members:
   :no-inherited-members:
   :show-inheritance:
