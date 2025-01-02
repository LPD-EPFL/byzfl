.. _server-label:

Server
=======

The ``Server`` class represents the central node in a federated learning simulation. It is responsible for aggregating updates from clients, maintaining the global model, and evaluating its performance. The server employs robust aggregation techniques to mitigate the impact of adversarial behavior introduced by Byzantine clients.

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

.. autoclass:: byzfl.Server
   :members:
   :undoc-members:
   :no-inherited-members:
   :show-inheritance:

Notes
-----
- The server is designed to be resilient against Byzantine behaviors by integrating pre-aggregation and aggregation techniques.
- Accuracy evaluation is built-in to monitor the model's progress throughout the simulation.
