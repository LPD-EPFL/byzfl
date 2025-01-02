.. _data-dist-label:

Data Distributor
================

The ``DataDistributor`` class is a utility for managing and distributing datasets among clients in a federated learning simulation. It supports various data distribution strategies to simulate realistic federated learning scenarios, including IID (independent and identically distributed) and non-IID distributions. This flexibility allows researchers to evaluate the performance and robustness of models under diverse data heterogeneity conditions.

Key Features
------------
- **Data Distribution Strategies**:
   - `iid`: Splits data equally and randomly among clients.
   - `gamma_similarity_niid`: Creates non-IID distributions with a specified degree of similarity using a `gamma` parameter.
   - `dirichlet_niid`: Implements non-IID distributions based on a Dirichlet distribution with a configurable concentration parameter (`alpha`).
   - `extreme_niid`: Generates highly non-IID distributions by assigning sorted data partitions to clients.

- **Flexible Input Support**: Accepts datasets as PyTorch `DataLoader` objects and processes them seamlessly.

- **Flexible Dataset Management**: Returns data loaders for each client after applying the specified data distribution.

.. autoclass:: byzfl.DataDistributor
   :members:
   :undoc-members:
   :no-inherited-members:
   :show-inheritance: