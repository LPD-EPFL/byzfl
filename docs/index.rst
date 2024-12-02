.. Byzantine Machine Learning Library Documentation documentation master file, created by
   sphinx-quickstart on Mon Mar 18 09:16:23 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
  :hidden:
  :titlesonly:
  
  aggregators/index
  attacker/index
  pipeline/index
  team/index



.. raw:: html

   <div style="visibility: hidden;height:1px;">
ByzFL Documentation
===================
.. raw:: html

   </div>
Welcome to the official documentation of ByzFL powered by 
`DCL <http://dcl.epfl.ch>`_ from `EPFL <http://epfl.ch>`_ and by
`WIDE <https://team.inria.fr/wide/>`_ from 
`INRIA Rennes <https://www.inria.fr/fr/centre-inria-universite-rennes>`_!

ByzFL is a Python Library for Byzantine-resilient Federated Learning 
compatible with `PyTorch <http://pytorch.org>`_ tensors and  `NumPy <http://numpy.org>`_ arrays.

ByzFL provides three main tools.

1. Robust aggregations and pre-aggregations.
2. Implementation of attacks.
3. Pipeline to train and benchmark new methods using ByzFL implemented schemes.

Our code is available on `Github <https://github.com/LPD-EPFL/byzfl>`_


.. rubric:: Getting Started

You can install the ByzFL module with pip command.

  >>> pip install byzfl

The `byzfl` module is then ready to use. For instance, to use the :ref:`mda-label` robust aggregation:

  >>> import byzfl
  >>> import numpy as np
  >>> 
  >>> agg = byzfl.MDA(1)
  >>> x = np.array([[1., 2., 3.],       # np.ndarray
  >>>               [4., 5., 6.],
  >>>               [7., 8., 9.]])
  >>> agg(x)
  array([2.5 3.5. 4.5])

Learn more about ByzFL:

.. raw:: html

  <div class="row">
  <div class="column">
    <div class="card">
      <div class="container">
        <h2>Aggregations</h2>
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

.. rubric:: Lisence

`MIT <https://github.com/LPD-EPFL/byzfl/blob/main/LICENSE.txt>`_

