.. _modelbaseinterface-label:

Model Base Interface
====================

The ``ModelBaseInterface`` class serves as an abstract interface for encapsulating models in the federated learning framework. It defines the necessary methods for managing model-related operations, ensuring consistent handling of parameters, gradients, and optimizers across derived classes.

Features
--------
- **Model Initialization**: Dynamically initializes a model based on its name, enabling seamless integration of various architectures.
- **Optimizer Flexibility**: Supports a variety of PyTorch optimizers, including SGD and Adam, allowing fine-grained control over training dynamics.
- **Gradient and Parameter Management**: Provides utilities for retrieving and setting model parameters and gradients in both flat and dictionary formats.
- **Scheduler Integration**: Includes a learning rate scheduler to handle milestone-based adjustments during training.

Initialization Parameters
-------------------------
All these parameters should be passed in a dictionary that contains the following keys:

- ``model_name`` (str): The name of the model to be used.
- ``device`` (str): The device for computation (e.g., 'cpu', 'cuda').
- ``optimizer_name`` (str): The name of the optimizer to use (e.g., "SGD", "Adam").
- ``optimizer_params`` (dict, optional): Additional parameters for the optimizer (e.g., beta values for Adam).
- ``learning_rate`` (float): Learning rate for the optimizer.
- ``weight_decay`` (float): Weight decay regularization parameter.
- ``milestones`` (list): List of epochs where the learning rate decay should be applied.
- ``learning_rate_decay`` (float): Multiplicative factor for learning rate decay.

Methods
-------
- ``get_flat_parameters``: Retrieves model parameters as a flat array.
- ``get_flat_gradients``: Retrieves model gradients as a flat array.
- ``get_dict_parameters``: Retrieves model parameters in dictionary format.
- ``get_dict_gradients``: Retrieves model gradients in dictionary format.
- ``set_parameters(flat_vector)``: Sets model parameters using a flat array.
- ``set_gradients(flat_vector)``: Sets model gradients using a flat array.
- ``set_model_state(state_dict)``: Updates the model's state using a state dictionary.

Usage Example
-------------
Below is an example of how to initialize the ``ModelBaseInterface`` and use its methods:

.. code-block:: python

   from byzfl import ModelBaseInterface

   params = {
       "model_name": "cnn_mnist",
       "device": "cpu",
       "optimizer_name": "Adam",
       "optimizer_params": {"betas": (0.9, 0.999)},
       "learning_rate": 0.01,
       "weight_decay": 0.0005,
       "milestones": [10, 20],
       "learning_rate_decay": 0.5
   }

   model_interface = ModelBaseInterface(params)

   # Access model parameters
   flat_params = model_interface.get_flat_parameters()

   # Update gradients
   model_interface.set_gradients(flat_vector=flat_params)

   # Perform an optimization step
   model_interface.optimizer.step()
   model_interface.scheduler.step()

.. autoclass:: byzfl.ModelBaseInterface
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:

Notes
-----
This class is not meant to be instantiated directly but serves as a base class for components such as the ``Client`` and ``Server`` classes in the federated learning framework.
