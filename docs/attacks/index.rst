.. _attacks-label:

Attacks
========

Welcome to the Byzantine Attacks module of the library, which provides implementations of various attack strategies targeting machine learning vectors, typically gradients. This module consists of two key components:

- Attack Implementation: Details the execution of specific attacks on vectors (or gradients) originating from honest participants.
- Attack Optimizer: Features an algorithm designed to identify the optimal parameters for an attack based on the given settings.

Explore these components to better understand and experiment with Byzantine attack strategies.

.. toctree::
   :caption: Attacks
   :titlesonly:

   classes/fall_of_empires
   classes/sign_flipping
   classes/little_is_enough
   classes/mimic
   classes/inf
   classes/gaussian

.. toctree::
   :caption: Attack optimizers
   :titlesonly:

   classes/line_maximize
   classes/worker_with_max_variance