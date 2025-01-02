.. _federated-learning-framework-label:

Federated Learning Framework
============================

The **Federated Learning Framework** provides a comprehensive environment for simulating and evaluating federated learning workflows. It integrates core components like ``Client``, ``Server``, ``ByzantineClient``, ``RobustAggregator``, and various ``Models`` to facilitate systematic experimentation and testing in distributed machine learning settings.

Features
--------

- **Simulate Real-World Federated Learning**: Recreate distributed learning scenarios involving multiple clients, a central server, and potential adversarial (Byzantine) participants.
- **Robust Aggregation**: Evaluate and compare aggregation strategies, incorporating pre-aggregation techniques such as ``Clipping`` and ``NNM`` with robust aggregators like ``TrMean``.
- **Byzantine Resilience**: Analyze the robustness of aggregation methods against malicious gradients introduced by Byzantine clients.
- **Flexibility and Extensibility**: Easily adapt to different datasets, models, and attack strategies, enabling extensive research and experimentation.

Purpose
-------

By leveraging this framework, researchers can gain valuable insights into the performance and resilience of aggregation methods under varying levels of dataset heterogeneity, numbers of adversaries, and other federated learning challenges. It serves as a powerful tool for advancing research in robust distributed machine learning.


.. toctree::
   :caption: Federated Learning Framework
   :titlesonly:

   classes/fed_learning_sim

.. toctree::
   :caption: Helper Modules
   :titlesonly:

   classes/robust_aggregators
   classes/client
   classes/byzantine_client
   classes/server
   classes/models
