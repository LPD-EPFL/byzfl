Aggregations
============

This is the Aggregation Module of the library, which contains all aggregation functionalities. In this module, you will find both aggregations and pre-aggregations. Aggregations combine updates from multiple participants aggregating them, while pre-aggregations perform preliminary calculations to reduce the data before the main aggregation, enhancing efficiency and robustness.


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

.. toctree::
   :caption: Pre-aggregators
   :titlesonly:

   classes/nnm
   classes/bucketing
   classes/clipping
   classes/arc


