.. Byzantine Machine Learning Library Documentation documentation master file, created by
   sphinx-quickstart on Mon Mar 18 09:16:23 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
  :hidden:
  :titlesonly:
  
  aggregators/index
  attacks/index
  fed_framework/index
  team/index


.. raw:: html

   <div style="visibility: hidden;height:1px;">
ByzFL Documentation
===================
.. raw:: html

   </div>
Welcome to the official documentation of ByzFL, developed by 
`DCL <http://dcl.epfl.ch>`_ from `EPFL <http://epfl.ch>`_ and
`WIDE <https://team.inria.fr/wide/>`_ from 
`INRIA Rennes <https://www.inria.fr/fr/centre-inria-universite-rennes>`_!

What is ByzFL?
==============

ByzFL is a **Python library for Byzantine-resilient Federated Learning**. It is designed to be fully compatible with both `PyTorch <http://pytorch.org>`_  tensors and `NumPy <http://numpy.org>`_ arrays, making it versatile for a wide range of machine learning workflows.

Key Features
------------

1. **Robust Aggregators and Pre-Aggregators**: Implement state-of-the-art aggregation strategies, such as :ref:`trmean-label`, alongside pre-aggregation techniques like :ref:`clipping-label` and :ref:`nnm-label`, to ensure resilience against Byzantine participants in federated learning workflows.
2. **Byzantine Attacks**: Simulate a wide range of adversarial attack scenarios, including :ref:`ipm-label` and :ref:`alie-label`, to rigorously test the robustness of federated learning systems under malicious conditions.
3. **Comprehensive Federated Learning Pipelines**: Seamlessly integrate key components — such as :ref:`client-label`, :ref:`server-label`, :ref:`byzantine-client-label`, and :ref:`data-dist-label` — to train, simulate, and benchmark robust aggregation schemes. The framework supports various datasets, models, and custom configurations, enabling researchers to replicate real-world distributed learning setups with heterogeneity and adversarial threats.


Installation
============

You can install the ByzFL library via pip:

.. code-block:: bash

   pip install byzfl


After installation, the library is ready to use. Here's a quick example of how to use the :ref:`trmean-label` robust aggregator and the :ref:`sf-label` Byzantine attack:


Quick Start Example - Using PyTorch Tensors
-------------------------------------------

.. code-block:: python

  import byzfl
  import torch

  # Number of Byzantine participants
  f = 1

  # Honest vectors
  honest_vectors = torch.tensor([[1., 2., 3.],
                                 [4., 5., 6.],
                                 [7., 8., 9.]])

  # Initialize and apply the attack
  attack = byzfl.SignFlipping()
  byz_vector = attack(honest_vectors)

  # Create f identical attack vectors
  byz_vectors = byz_vector.repeat(f, 1)

  # Concatenate honest and Byzantine vectors
  all_vectors = torch.cat((honest_vectors, byz_vectors), dim=0)

  # Initialize and perform robust aggregation
  aggregate = byzfl.TrMean(f=f)
  result = aggregate(all_vectors)
  print("Aggregated result:", result)

Output:

.. code-block:: none

   Aggregated result: tensor([2.5000, 3.5000, 4.5000])


Quick Start Example - Using NumPy Arrays
----------------------------------------

.. code-block:: python

   import byzfl
   import numpy as np

   # Number of Byzantine participants
   f = 1

   # Honest vectors
   honest_vectors = np.array([[1., 2., 3.],
                              [4., 5., 6.],
                              [7., 8., 9.]])

   # Initialize and apply the attack
   attack = byzfl.SignFlipping()
   byz_vector = attack(honest_vectors)

   # Create f identical attack vectors
   byz_vectors = np.tile(byz_vector, (f, 1))

   # Concatenate honest and Byzantine vectors
   all_vectors = np.concatenate((honest_vectors, byz_vectors), axis=0)

   # Initialize and perform robust aggregation
   aggregate = byzfl.TrMean(f=f)
   result = aggregate(all_vectors)
   print("Aggregated result:", result)

Output:

.. code-block:: none

   Aggregated result: [2.5 3.5 4.5]

Learn More
==========

Explore the key components of ByzFL below:

.. raw:: html

  <div class="row">
  <div class="column">
    <div class="card">
      <div class="container">
        <h2>Aggregators</h2>
        <p>

:ref:`Get Started <aggregations-label>`

.. raw:: html

        </p>
      </div>
    </div>
  </div>

  <div class="column">
    <div class="card">
      <div class="container">
        <h2>Attacks</h2>

:ref:`Get Started <attacks-label>`

.. raw:: html

      </div>
    </div>
  </div>

  <div class="column">
    <div class="card">
      <div class="container">
        <h2>FL Framework</h2>

:ref:`Get Started <federated-learning-framework-label>`

.. raw:: html

      </div>
    </div>
  </div>
  </div>

License
=======

ByzFL is open-source and distributed under the `MIT License <https://github.com/LPD-EPFL/byzfl/blob/main/LICENSE.txt>`_.

Our code is hosted on `Github <https://github.com/LPD-EPFL/byzfl>`_. Contributions are welcome!