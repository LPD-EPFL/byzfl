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

Spiking Models
--------------

The following models use snnTorch leaky integrate-and-fire neurons and declare
``is_snn = True``. snnTorch is a required dependency, listed in
``requirements.txt``.

.. list-table:: Available SNN architectures
   :header-rows: 1

   * - Model
     - Default input per sample and time step
     - Architecture
   * - ``fc_snn``
     - 784 features, or a 1 x 28 x 28 image
     - Fully connected layers with 100 hidden neurons, matching ``fc_mnist``
   * - ``cnn_mnist_snn``
     - 1 x 28 x 28
     - 20/50 convolutional filters and 500 hidden neurons
   * - ``cnn_cifar_snn``
     - 3 x 32 x 32
     - 20/100/200 convolutional filters and 512/256 hidden neurons

All models default to 10 output classes; set ``output_dim`` to change this.

Inputs, outputs, and state
~~~~~~~~~~~~~~~~~~~~~~~~~~

Convolutional SNNs accept encoded images with shape
``(batch, time, channels, height, width)``. ``fc_snn`` also accepts encoded
vectors with shape ``(batch, time, features)``.

Use ``TemporalEncoder`` to encode static inputs before calling a model directly.
All encoding modes provide an explicit time dimension. Constant encoding uses
an expanded view without copying the input across time. Models derive sequence
length from their input and do not accept a ``time_steps`` constructor argument.
In benchmark configuration, specify duration in ``model.encoding.time_steps``.

Every forward pass returns ``(spikes, membrane)``, with both tensors shaped
``(time, batch, output_dim)``. Both remain available for custom loss functions,
accuracy functions, and analysis. Each forward pass resets its own neuron
states, so unrelated batches do not share membrane history.

.. code-block:: python

   import torch
   from byzfl import cnn_mnist_snn
   from byzfl.fed_framework.encoding import TemporalEncoder

   model = cnn_mnist_snn(
       beta=0.95,
       surrogate_gradient="atan",
       surrogate_params={"alpha": 1.2},
   )
   encoder = TemporalEncoder(time_steps=10, encoding_type="constant")
   spikes, membrane = model(encoder(torch.rand(2, 1, 28, 28)))
   assert spikes.shape == membrane.shape == (10, 2, 10)

Surrogate gradients
~~~~~~~~~~~~~~~~~~~

``surrogate_gradient`` selects a factory by name and ``surrogate_params``
passes its keyword arguments. The default is ``"atan"`` with the snnTorch
factory's defaults. Other named snnTorch factories, such as ``"fast_sigmoid"``
and ``"triangular"``, are resolved through
``byzfl/fed_framework/surrogates.py:get_spike_grad``.

The callable passed as ``spike_grad`` to each snnTorch neuron generates binary
spikes in the forward pass and supplies a surrogate derivative for training.
``get_spike_grad`` only resolves and configures this callable; it does not
calculate gradients itself. Omitting ``surrogate_params`` uses the factory's
defaults.

Adding a custom surrogate
^^^^^^^^^^^^^^^^^^^^^^^^^

1. Add a factory to ``byzfl/fed_framework/surrogates.py``. The factory accepts
   the configurable parameters and returns a callable produced by snnTorch's
   ``custom_surrogate`` helper:

.. code-block:: python

   from snntorch import surrogate

   def my_custom_surrogate(scale=1.0):
       def custom_gradient(input_, grad_input, spikes):
           return grad_input * scale / (1 + input_.square())

       return surrogate.custom_surrogate(custom_gradient)

2. Add the configuration name and factory to ``CUSTOM_SURROGATES``:

.. code-block:: python

   CUSTOM_SURROGATES["MyCustomSurrogate"] = my_custom_surrogate

3. Select that name under ``model.model_params`` in your existing configuration:

.. code-block:: json

   {
       "model": {
           "model_params": {
               "surrogate_gradient": "MyCustomSurrogate",
               "surrogate_params": {"scale": 0.5}
           }
       }
   }

``get_spike_grad`` checks ``CUSTOM_SURROGATES`` and then the factories provided
by ``snntorch.surrogate``. Both use the same interface: ``surrogate_params`` are
passed to the factory as keyword arguments, and the factory must return a
callable that accepts the neuron input. Custom names must be distinct from
snnTorch names; collisions raise an error. ``MyCustomSurrogate`` is a
documentation example and is not registered by default.

Unknown constructor arguments and surrogate parameters raise errors, so
misspellings cannot silently change an experiment. ``threshold`` and
``learn_threshold`` are supported by ``cnn_mnist_snn`` and ``cnn_cifar_snn``.
``fc_snn`` uses the default neuron threshold.

Adding a custom spiking model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add the model class to ``byzfl/fed_framework/models.py``. Model lookup uses the
class name directly, so no registry or lookup function must be edited. A custom
spiking model must:

* inherit from ``torch.nn.Module``;
* declare the class attribute ``is_snn = True``;
* accept its configurable values as constructor keyword arguments;
* accept temporal inputs and derive sequence length from ``inputs.size(1)``; and
* return ``(spikes, membrane)``, each shaped ``(time, batch, classes)``.

For example, this declaration makes ``"MySpikingModel"`` a valid model name:

.. code-block:: python

   class MySpikingModel(torch.nn.Module):
       is_snn = True

       def __init__(self, output_dim=10):
           super().__init__()
           # Define layers and spiking neurons here.

       def forward(self, inputs):
           # Return tensors shaped (time, batch, output_dim).
           return spikes, membrane

The class declaration is authoritative. An optional ``model.is_snn`` value in
benchmark JSON only checks that declaration and cannot convert an ANN into an SNN.
Classes without ``is_snn`` retain ANN behavior. Constructor misspellings raise an
error instead of being ignored for SNN models.

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

.. autoclass:: byzfl.fc_snn
   :members:
   :no-inherited-members:

.. autoclass:: byzfl.cnn_mnist_snn
   :members:
   :no-inherited-members:

.. autoclass:: byzfl.cnn_cifar_snn
   :members:
   :no-inherited-members:
