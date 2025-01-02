.. _client-label:

Client
=======

The ``Client`` class simulates an honest participant in a federated learning environment. Each client trains its local model on its private dataset and shares its updates (gradients) with the server for aggregation. The class is designed to integrate seamlessly into the federated learning pipeline provided by the ByzFL framework.

Key Features
------------
- **Local Training**: Allows training on client-specific datasets while maintaining data ownership.
- **Gradient Computation**: Computes gradients of the model's loss function with respect to its parameters.
- **Support for Momentum**: Incorporates momentum into gradient updates to improve convergence.
- **Integration with Robust Aggregators**: Shares updates with the server, enabling robust aggregation techniques to handle adversarial or heterogeneous data environments.

.. autoclass:: byzfl.Client
   :members:
   :undoc-members:
   :no-inherited-members:
   :show-inheritance: