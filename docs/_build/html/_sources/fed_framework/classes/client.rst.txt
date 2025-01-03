.. _client-label:

Client
======

The ``Client`` class simulates an honest participant in a federated learning environment. Each client trains its local model on its subset of the data and shares updates (e.g., gradients) with the central server.

Key Features
------------
- **Local Training**: Allows training on client-specific datasets while maintaining data ownership.
- **Gradient Computation**: Computes gradients of the model's loss function with respect to its parameters.
- **Support for Momentum**: Incorporates momentum into gradient updates to improve convergence.
- **Integration with Robust Aggregators**: Shares updates with the server, enabling robust aggregation techniques to handle adversarial or heterogeneous data environments.

Initialization Parameters
-------------------------
:param dict params:
    A dictionary containing the configuration for the Client. Must include:

- ``"model_name"`` : str
   Name of the model to be used. For a complete list of available models, refer to :ref:`models-label`.
- ``"device"`` : str
   Device for computation (e.g., ``'cpu'`` or ``'cuda'``).
- ``"optimizer_name"`` : str 
   Name of the optimizer to be used (e.g., ``"SGD"``, ``"Adam"``).
- ``"optimizer_params"`` : dict, optional 
   Additional parameters for the optimizer (e.g., beta values for Adam).
- ``"learning_rate"`` : float 
   Learning rate for the optimizer.
- ``"loss_name"`` : str
   Name of the loss function to be used (e.g., ``"CrossEntropyLoss"``).
- ``"weight_decay"`` : float 
   Weight decay for regularization.
- ``"milestones"`` : list 
   List of training steps at which the learning rate decay is applied.
- ``"learning_rate_decay"`` : float 
   Factor by which the learning rate is reduced at each milestone.
- ``"LabelFlipping"`` : bool 
   A flag that enables the label-flipping attack. When set to ``True``, the class labels in the local dataset are flipped to their opposing classes.
- ``"momentum"`` : float 
   Momentum parameter for the optimizer.
- ``"training_dataloader"`` : DataLoader 
   PyTorch DataLoader object for the local training dataset.
- ``"nb_labels"`` : int 
   Number of labels in the dataset, required for the label-flipping attack.

Methods
-------

- ``compute_gradients()``
  Computes the gradients of the local model based on the client's subset of training data.

- ``get_flat_gradients_with_momentum()``
  Returns the flattened gradients with momentum applied, combining current and past updates.

- ``get_flat_flipped_gradients()``
  Retrieves the gradients of the model after applying the label-flipping attack in a flattened array.

- ``get_loss_list()``
  Returns the list of training losses recorded during training.

- ``get_train_accuracy()``
  Provides the training accuracy for each processed batch.

- ``set_model_state(state_dict)``
  Updates the model's state with the provided state dictionary.


Examples
--------
Initialize the ``Client`` class with an MNIST data loader:

.. code-block:: python

    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from byzfl import Client

    # Fix the random seed
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Define the training data loader using MNIST
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Define client parameters
    client_params = {
        "model_name": "cnn_mnist",
        "device": "cpu",
        "optimizer_name": "Adam",
        "optimizer_params": {"betas": (0.9, 0.999)},
        "learning_rate": 0.01,
        "loss_name": "CrossEntropyLoss",
        "weight_decay": 0.0005,
        "milestones": [10, 20],
        "learning_rate_decay": 0.5,
        "LabelFlipping": True,
        "momentum": 0.9,
        "training_dataloader": train_loader,
        "nb_labels": 10,
    }

    # Initialize the Client
    client = Client(client_params)

Compute gradients for the local dataset:

.. code-block:: python

    # Compute gradients
    client.compute_gradients()

    # Retrieve the training accuracy for the first batch
    print(client.get_train_accuracy()[0])

    # Retrieve the gradients after applying the label-flipping attack
    print(client.get_flat_flipped_gradients())


.. autoclass:: byzfl.Client
   :members:
   :undoc-members:
   :no-inherited-members:
   :show-inheritance: