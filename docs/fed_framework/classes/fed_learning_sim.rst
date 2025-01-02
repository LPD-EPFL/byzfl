.. _federated_learning-label:

Federated Learning Simulation
=============================

The **Federated Learning Simulation** module demonstrates how to use the key components of the library — ``Client``, ``Server``, ``ByzantineClient``, and ``DataDistributor`` — to simulate a federated learning environment. This example showcases how to perform distributed learning with Byzantine-resilient aggregation strategies.

Key Features
------------
- **Client Class**: Simulates honest participants that compute gradients based on their local data.
- **Server Class**: Simulates the central server that aggregates gradients and updates the global model.
- **ByzantineClient Class**: Simulates malicious participants injecting adversarial gradients into the aggregation process.
- **DataDistributor Class**: Handles the distribution of data among clients in various configurations, including IID and non-IID distributions (e.g., Dirichlet, Gamma, Extreme), to simulate realistic federated learning setups.
- **Robust Aggregation**: Demonstrates the usage of robust aggregation techniques such as :ref:`trmean-label`, combined with pre-aggregation methods like :ref:`clipping-label` and :ref:`nnm-label`.

Example: Federated Learning Workflow
------------------------------------
This example uses the MNIST dataset to simulate a federated learning setup with five honest clients and two Byzantine clients. Follow the steps below to run the simulation:

.. code-block:: python

    # Import necessary libraries
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from byzfl import Client, Server, ByzantineClient, DataDistributor

    # Set random seed for reproducibility
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Configurations
    nb_honest_clients = 5
    nb_byz_clients = 2
    nb_training_steps = 10
    batch_size = 64

    # Data Preparation
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
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
            "learning_rate": 0.1,
            "loss_name": "NLLLoss",
            "weight_decay": 0.0005,
            "milestones": [10, 20],
            "learning_rate_decay": 0.25,
            "LabelFlipping": False,
            "training_dataloader": client_dataloaders[i],
            "momentum": 0.9,
            "nb_labels": 10,
        }) for i in range(nb_honest_clients)
    ]

    # Prepare Test Dataset
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Server Setup
    pre_aggregators = [
        {"name": "Clipping", "parameters": {"c": 2.0}},
        {"name": "NNM", "parameters": {"f": nb_byz_clients}},
    ]
    aggregator_info = {"name": "TrMean", "parameters": {"f": nb_byz_clients}}
    server = Server({
        "device": "cpu",
        "model_name": "cnn_mnist",
        "test_loader": test_loader,
        "learning_rate": 0.1,
        "weight_decay": 0.0005,
        "milestones": [10, 20],
        "learning_rate_decay": 0.25,
        "aggregator_info": aggregator_info,
        "pre_agg_list": pre_aggregators,
    })

    # Byzantine Client Setup
    attack = {
        "name": "InnerProductManipulation",
        "f": nb_byz_clients,
        "parameters": {"tau": 3.0},
    }
    byz_client = ByzantineClient(attack)

    # Training Loop
    for training_step in range(nb_training_steps):
        print(f"--- Training Step {training_step + 1}/{nb_training_steps} ---")

        # Honest Clients Compute Gradients
        for client in honest_clients:
            client.compute_gradients()

        # Aggregate Honest Gradients
        honest_gradients = [client.get_flat_gradients_with_momentum() for client in honest_clients]

        # Apply Byzantine Attack
        byz_gradients = byz_client.apply_attack(honest_gradients)

        # Combine Honest and Byzantine Gradients
        gradients = honest_gradients + byz_gradients

        # Robustly Aggregate Gradients and Update Global Model
        server.update_model(gradients)

        # Evaluate Global Model
        test_acc = server.compute_test_accuracy()
        print(f"Test Accuracy: {test_acc:.4f}")

    print("Training Complete!")

Example Output
--------------
Running the above code will produce the following output:

.. code-block:: text

    --- Training Step 1/10 ---
    Test Accuracy: 0.1012
    --- Training Step 2/10 ---
    Test Accuracy: 0.1011
    --- Training Step 3/10 ---
    Test Accuracy: 0.1010
    --- Training Step 4/10 ---
    Test Accuracy: 0.1010
    --- Training Step 5/10 ---
    Test Accuracy: 0.1010
    --- Training Step 6/10 ---
    Test Accuracy: 0.1010
    --- Training Step 7/10 ---
    Test Accuracy: 0.1010
    --- Training Step 8/10 ---
    Test Accuracy: 0.1010
    --- Training Step 9/10 ---
    Test Accuracy: 0.1010
    --- Training Step 10/10 ---
    Test Accuracy: 0.1010
    Training Complete!

Documentation References
------------------------
For more information about individual components, refer to the following:
- **Client Class**: :ref:`client-label`
- **Server Class**: :ref:`server-label`
- **ByzantineClient Class**: :ref:`byzantine-client-label`
- **RobustAggregator Class**: :ref:`robust-aggregator-label`
- **DataDistributor Class**: :ref:`data-dist-label`
- **Models Module**: :ref:`models-label`

Notes
-----
- This example can be extended to other datasets and models by modifying the parameters accordingly.
- The robustness of the system depends on the aggregation methods and the number of Byzantine participants.
- The module is designed to be flexible and adaptable for experimentation with different setups.
