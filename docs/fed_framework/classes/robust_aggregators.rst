.. _robust-aggregator-label:

Robust Aggregator
=================

The ``RobustAggregator`` class provides a robust mechanism for aggregating gradients in federated learning setups. It combines pre-aggregation techniques with robust aggregation methods to mitigate the effects of adversarial inputs and outliers, ensuring reliable updates for the global model.

Key Features
------------
- **Pre-Aggregation**:  
  Supports applying multiple pre-aggregation techniques sequentially, such as :ref:`clipping-label` and :ref:`nnm-label`, to refine the input gradients.
  
- **Robust Aggregation**:  
  Implements aggregation strategies like :ref:`trmean-label` to handle Byzantine gradients and ensure robustness against adversarial attacks.
  
- **Flexible Input Handling**:  
  Compatible with various input formats, including NumPy arrays, PyTorch tensors, and lists of these data types.

.. autoclass:: byzfl.RobustAggregator
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:

Notes
-----
- The pre-aggregation methods are applied in the order they are listed in `pre_agg_list`.
- The robust aggregation method is applied after all pre-aggregation techniques are completed.
