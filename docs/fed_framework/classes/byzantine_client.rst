.. _byzantine-client-label:

Byzantine Client
================

The ``ByzantineClient`` class simulates malicious participants in a federated learning setup. These clients perform adversarial attacks to disrupt the training process by introducing corrupted updates, enabling the evaluation of robust aggregation methods and defenses against Byzantine behavior.

Key Features
------------
- **Simulated Attacks**:  
  Implements various attack strategies, such as :ref:`ipm-label` and :ref:`alie-label`, to evaluate the robustness of aggregation techniques.

- **Configurable Behavior**:  
  Allows customization of the type and intensity of attacks through a flexible parameterization system.

- **Integration with Federated Learning**:  
  Easily integrates into federated learning workflows alongside honest clients and the central server.

.. autoclass:: byzfl.ByzantineClient
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:

Notes
-----
- The number of adversarial vectors generated is determined by the `"f"` parameter.
- Attack strategies can be extended or modified by customizing the attack implementation in the library.
