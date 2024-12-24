.. Byzantine Machine Learning Library Documentation documentation master file, created by
   sphinx-quickstart on Mon Mar 18 09:16:23 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
  :hidden:
  :titlesonly:
  
  aggregators/index
  attacks/index
  pipeline/index
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

1. **Robust Aggregators and Pre-Aggregators**: Robustly aggregate gradients while mitigating the impact of Byzantine participants.
2. **Byzantine Attacks**: Simulate and evaluate different attack strategies for testing resilience.
3. **ML Pipelines**: Train and benchmark robust aggregation schemes and attack implementations seamlessly.


Installation
============

You can install the ByzFL library via pip:

.. code-block:: bash

   pip install byzfl


After installation, the library is ready to use. Here's a quick example of how to use the :ref:`trmean-label` robust aggregator and the :ref:`sf-label` Byzantine attack:


Quick Start Example
------------

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
        <h2>Pipeline</h2>

:ref:`Get Started <pipeline-label>`

.. raw:: html

      </div>
    </div>
  </div>
  </div>

License
=======

ByzFL is open-source and distributed under the `MIT License <https://github.com/LPD-EPFL/byzfl/blob/main/LICENSE.txt>`_.

Our code is hosted on `Github <https://github.com/LPD-EPFL/byzfl>`_. Contributions are welcome!