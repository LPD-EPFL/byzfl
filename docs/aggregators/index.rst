.. _aggregations-label:

Aggregators
============

Welcome to the Aggregation module of the library, which provides a comprehensive suite of aggregation functionalities. This module includes both aggregators and pre-aggregators:

- Aggregators: Combine updates from multiple participants by aggregating their inputs.
- Pre-aggregators: Perform preliminary computations to transform the data before the main aggregation, enhancing robustness.

Explore the available aggregation methods below:

.. toctree::
   :caption: Aggregators
   :titlesonly:

   classes/average
   classes/median
   classes/trmean
   classes/geometric_median
   classes/krum
   classes/multi_krum
   classes/centered_clipping
   classes/mda
   classes/monna
   classes/meamed
   classes/caf
   classes/smea

.. toctree::
   :caption: Pre-aggregators
   :titlesonly:

   classes/nnm
   classes/bucketing
   classes/clipping
   classes/arc