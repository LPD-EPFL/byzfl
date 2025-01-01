.. _federated_learning-label:

Federated Learning Simulation
=============================

The **Federated Learning Simulation** module demonstrates how to use the key components of the library—`Client`, `Server`, and `ByzantineClient`—to simulate a federated learning environment. This example showcases how to perform distributed learning with Byzantine-resilient aggregation strategies.

Key Features
------------
- **Client Class**: Simulates honest participants that compute gradients based on their local data.
- **Server Class**: Simulates the central server that aggregates gradients and updates the global model.
- **ByzantineClient Class**: Simulates malicious participants injecting adversarial gradients into the aggregation process.
- **Robust Aggregation**: Demonstrates the usage of robust aggregation techniques such as :ref:`trmean-label`, combined with pre-aggregation methods like :ref:`clipping-label` and :ref:`nnm-label`.

Example: Federated Learning Workflow
------------------------------------
This example uses the MNIST dataset to simulate a federated learning setup with multiple honest clients and one Byzantine client. Follow the steps below to run the simulation:

.. code-block:: python

   # Import necessary libraries
   import torch
   from torch.utils.data import DataLoader
   from torchvision import datasets, transforms
   from byzfl import Client, Server, ByzantineClient

   # Set random seed for reproducibility
   SEED = 42
   torch.manual_seed(SEED)
   torch.cuda.manual_seed(SEED)
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False

   # Prepare the MNIST dataset
   transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
   train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
   train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

   test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
   test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

   # Define client parameters
   client_params = {
       "model_name": "cnn_mnist",
       "device": "cpu",
       "learning_rate": 0.01,
       "loss_name": "CrossEntropyLoss",
       "weight_decay": 0.0005,
       "milestones": [10, 20],
       "learning_rate_decay": 0.5,
       "LabelFlipping": True,
       "momentum": 0.9,
       "training_dataloader": train_loader,
       "nb_labels": 10,
       "nb_steps": len(train_loader),
   }

   # Define pre-aggregation methods
   pre_aggregators = [
       {"name": "Clipping", "parameters": {"c": 2.0}},
       {"name": "NNM", "parameters": {"f": 1}},
   ]

   # Define aggregation method
   aggregator_info = {"name": "TrMean", "parameters": {"f": 1}}

   # Define server parameters
   server_params = {
       "device": "cpu",
       "model_name": "cnn_mnist",
       "test_loader": test_loader,
       "validation_loader": None,
       "learning_rate": 0.01,
       "weight_decay": 0.0005,
       "milestones": [10, 20],
       "learning_rate_decay": 0.5,
       "aggregator_info": aggregator_info,
       "pre_agg_list": pre_aggregators,
   }

   # Initialize the Server
   server = Server(server_params)

   # Initialize the Clients
   client1 = Client(client_params)
   client2 = Client(client_params)
   client3 = Client(client_params)
   clients = [client1, client2, client3]

   # Define a Byzantine Client
   attack = {
       "name": "InnerProductManipulation",
       "f": 1,
       "parameters": {"tau": 3.0},
   }
   byz_client = ByzantineClient(attack)

   # Training loop
   for epoch in range(1, 3):  # Simulate 2 epochs
       print(f"Epoch {epoch}")
       # Clients compute their gradients
       for client in clients:
           client.compute_gradients()

       # Collect gradients from honest clients
       honest_gradients = [client.get_flat_gradients() for client in clients]

       # Apply Byzantine attack
       byz_vector = byz_client.apply_attack(honest_gradients)

       # Combine honest gradients and Byzantine gradients
       gradients = honest_gradients + byz_vector

       # Server aggregates gradients and updates the global model
       server.update_model(gradients)

       # Evaluate global model
       test_acc = server.compute_test_accuracy()
       print(f"Test Accuracy: {test_acc:.4f}")

Example Output
--------------
Running the above code will produce the following output:

.. code-block:: text

   Epoch 1
   Test Accuracy: 0.1013
   Epoch 2
   Test Accuracy: 0.1016

Documentation References
------------------------
For more information about individual components, refer to the following:
- **Client Class**: :ref:`client-label`
- **Server Class**: :ref:`server-label`
- **ByzantineClient Class**: :ref:`byzantine-client-label`
- **RobustAggregator Class**: :ref:`robust-aggregator-label`
- **Models Module**: :ref:`models-label`

Notes
-----
- This example can be extended to other datasets and models by modifying the parameters accordingly.
- The robustness of the system depends on the aggregation methods and the number of Byzantine participants.
- The module is designed to be flexible and adaptable for experimentation with different setups.
