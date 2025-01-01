.. _models-label:

Models
======

The Models module provides a collection of neural network architectures for use with the ``Client`` and ``Server`` classes in the Federated Learning Framework. These models include fully connected networks, convolutional networks, logistic regression, and ResNet architectures, tailored for datasets such as MNIST, CIFAR-10, CIFAR-100, and ImageNet.

Available Models
----------------

**MNIST Models**
~~~~~~~~~~~~~~~~~

- **`fc_mnist`**: Fully connected neural network for MNIST.
- **`cnn_mnist`**: Convolutional neural network for MNIST.
- **`logreg_mnist`**: Logistic regression model for MNIST.

**CIFAR Models**
~~~~~~~~~~~~~~~~~

- **`cnn_cifar`**: Convolutional neural network for CIFAR datasets.

**ResNet Models**
~~~~~~~~~~~~~~~~~~

- **`ResNet18`**: ResNet with 18 layers.
- **`ResNet34`**: ResNet with 34 layers.
- **`ResNet50`**: ResNet with 50 layers.
- **`ResNet101`**: ResNet with 101 layers.
- **`ResNet152`**: ResNet with 152 layers.

Usage Examples
--------------

Each model can be easily imported and used in your training framework. For instance:

Using `ResNet18` for CIFAR-10:

.. code-block:: python

   from models import ResNet18
   model = ResNet18(num_classes=10)
   print(model)

Using `fc_mnist` for MNIST:

.. code-block:: python

   from models import fc_mnist
   model = fc_mnist()
   print(model)

.. note::

   A detailed description of these models, including their architecture and intended use, can be found below.

API Documentation
------------------

.. autoclass:: byzfl.fc_mnist
   :members:
   :undoc-members:
   :no-inherited-members:
   :show-inheritance:

.. autoclass:: byzfl.cnn_mnist
   :members:
   :undoc-members:
   :no-inherited-members:
   :show-inheritance:

.. autoclass:: byzfl.logreg_mnist
   :members:
   :undoc-members:
   :no-inherited-members:
   :show-inheritance:

.. autoclass:: byzfl.cnn_cifar
   :members:
   :undoc-members:
   :no-inherited-members:
   :show-inheritance:

.. autoclass:: byzfl.ResNet18
   :members:
   :undoc-members:
   :no-inherited-members:
   :show-inheritance:

.. autoclass:: byzfl.ResNet34
   :members:
   :undoc-members:
   :no-inherited-members:
   :show-inheritance:

.. autoclass:: byzfl.ResNet50
   :members:
   :undoc-members:
   :no-inherited-members:
   :show-inheritance:

.. autoclass:: byzfl.ResNet101
   :members:
   :undoc-members:
   :no-inherited-members:
   :show-inheritance:

.. autoclass:: byzfl.ResNet152
   :members:
   :undoc-members:
   :no-inherited-members:
   :show-inheritance:
