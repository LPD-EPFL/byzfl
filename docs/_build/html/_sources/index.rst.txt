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
Welcome to the official documentation of ByzFL powered by 
`DCL <http://dcl.epfl.ch>`_ from `EPFL <http://epfl.ch>`_ and by
`WIDE <https://team.inria.fr/wide/>`_ from 
`INRIA Rennes <https://www.inria.fr/fr/centre-inria-universite-rennes>`_!

ByzFL is a Python Library for Byzantine-resilient Federated Learning 
compatible with `PyTorch <http://pytorch.org>`_ tensors and  `NumPy <http://numpy.org>`_ arrays.

ByzFL provides three main tools.

1. Robust aggregators and pre-aggregators.
2. Implementation of Byzantine attacks.
3. Pipeline to train and benchmark new methods using ByzFL implemented schemes.

Our code is available on `Github <https://github.com/LPD-EPFL/byzfl>`_.


.. rubric:: Getting Started

You can install the ByzFL module using pip.

  >>> pip install byzfl

The `byzfl` module is then ready to use. For instance, the following is an example using the :ref:`trmean-label` robust aggregator and the :ref:`sf-label` Byzantine attack:

  >>> import byzfl
  >>> import numpy as np
  >>> f = 1                                                 # Number of Byzantine participants
  >>>
  >>> honest_vectors = np.array([[1., 2., 3.],              # Honest vectors
  >>>                            [4., 5., 6.],
  >>>                            [7., 8., 9.]])
  >>>
  >>> attack = byzfl.SignFlipping()                         # Initialize the attack
  >>> byz_vector = attack(honest_vectors)                   # Generate a single attack vector
  >>>
  >>> # Create f identical attack vectors
  >>> byz_vectors = np.tile(byz_vector, (f, 1))
  >>>
  >>> # Concatenate honest and Byzantine vectors
  >>> all_vectors = np.concatenate((honest_vectors, byz_vectors), axis=0)
  >>>
  >>> aggregate = byzfl.Trmean(f=f)
  >>> aggregate(all_vectors)                                # Perform robust aggregation on all vectors
  array([2.5 3.5 4.5])

Learn more about ByzFL:

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

.. rubric:: Lisence

`MIT <https://github.com/LPD-EPFL/byzfl/blob/main/LICENSE.txt>`_