.. _federated_learning-sim-label:

FL Simulation
==================================

The **FL Simulation** module demonstrates how to use the key components of the library — ``Client``, ``Server``, ``ByzantineClient``, and ``DataDistributor`` — to simulate a federated learning environment. This example showcases how to perform distributed learning with Byzantine-resilient aggregation strategies.

Key Features
------------
- ``Client`` Class: Simulates honest participants that compute gradients based on their local data.
- ``Server`` Class: Simulates the central server that aggregates gradients and updates the global model.
- ``ByzantineClient`` Class: Simulates malicious participants injecting adversarial gradients into the aggregation process.
- ``DataDistributor`` Class: Handles the distribution of data among clients in various configurations, including IID and non-IID distributions (e.g., Dirichlet, Gamma, Extreme), to simulate realistic federated learning setups.
- ``RobustAggregator`` Class: Demonstrates the usage of robust aggregation techniques such as :ref:`trmean-label`, combined with pre-aggregation methods like :ref:`clipping-label` and :ref:`nnm-label`.
- ``Models`` Module: Stores a collection of available ML models for federated learning tasks on MNIST and CIFAR-10.

Example: Federated Learning Workflow
------------------------------------
This example uses the MNIST dataset to simulate a federated learning setup with five honest clients and two Byzantine clients. Follow the steps below to run the simulation:

.. code-block:: python

    # Import necessary libraries
    from torch import Tensor
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
    train_dataset.targets = Tensor(train_dataset.targets).long() # Required for Datasets that targets are not Tensor (e.g. CIFAR-10)
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
            "optimizer_name": "SGD",
            "learning_rate": 0.1,
            "loss_name": "NLLLoss",
            "weight_decay": 0.0001,
            "milestones": [1000],
            "learning_rate_decay": 0.25,
            "LabelFlipping": False,
            "training_dataloader": client_dataloaders[i],
            "momentum": 0.9,
            "nb_labels": 10,
        }) for i in range(nb_honest_clients)
    ]

    # Prepare Test Dataset
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_dataset.targets = Tensor(test_dataset.targets).long() # Required for Datasets that targets are not Tensor (e.g. CIFAR-10)
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

    # Send Initial Model to All Clients
    new_model = server.get_dict_parameters()
    for client in honest_clients:
        client.set_model_state(new_model)

    # Training Loop
    for training_step in range(nb_training_steps):

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

        # Send Updated Model to Clients
        new_model = server.get_dict_parameters()
        for client in honest_clients:
            client.set_model_state(new_model)

    print("Training Complete!")

Example Output
--------------
Running the above code will produce the following output:

.. code-block:: text

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

Notes
-----
- This example can be extended to other datasets and models by modifying the parameters accordingly.
- The robustness of the system depends on the aggregation methods and the number of Byzantine participants.
- The module is designed to be flexible and adaptable for experimentation with different setups.

Documentation References
------------------------
For more information about individual components, refer to the following:
    - ``Client`` Class: :ref:`client-label`
    - ``Server`` Class: :ref:`server-label`
    - ``ByzantineClient`` Class: :ref:`byzantine-client-label`
    - ``RobustAggregator`` Class: :ref:`robust-aggregator-label`
    - ``DataDistributor`` Class: :ref:`data-dist-label`
    - ``Models`` Module: :ref:`models-label`