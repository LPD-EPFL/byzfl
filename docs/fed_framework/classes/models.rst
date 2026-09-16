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

Convolutional SNNs accept static images with shape ``(batch, channels, height,
width)`` or encoded images with shape ``(batch, time, channels, height, width)``.
``fc_snn`` also accepts static vectors ``(batch, features)`` and encoded vectors
``(batch, time, features)``.

Static inputs are repeated without copying over the constructor's
``time_steps`` (default: 25). Already encoded inputs retain their supplied
sequence length. This repetition does not perform rate or latency encoding.
In benchmark configuration, specify time steps only in
``model.encoding.time_steps``; the constructor argument is the internal way
to supply that value to the model.

Every forward pass returns ``(spikes, membrane)``, with both tensors shaped
``(time, batch, output_dim)``. Both remain available for custom loss functions,
accuracy functions, and analysis. Each forward pass resets its own neuron
states, so unrelated batches do not share membrane history.

.. code-block:: python

   import torch
   from byzfl import cnn_mnist_snn

   model = cnn_mnist_snn(
       time_steps=10,
       beta=0.95,
       surrogate_gradient="atan",
       surrogate_params={"alpha": 1.2},
   )
   spikes, membrane = model(torch.rand(2, 1, 28, 28))
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

1. Add a ``torch.autograd.Function`` subclass directly to
   ``byzfl/fed_framework/surrogates.py``, alongside ``get_spike_grad``. Define
   the spike operation in ``forward`` and its surrogate derivative in
   ``backward``. Copy and adapt this example:

.. code-block:: python

   import torch

   class MyCustomSurrogate(torch.autograd.Function):
       @staticmethod
       def forward(ctx, input_, scale=1.0):
           ctx.save_for_backward(input_)
           ctx.scale = scale
           return (input_ > 0).float()

       @staticmethod
       def backward(ctx, grad_output):
           (input_,) = ctx.saved_tensors
           return grad_output * ctx.scale / (1 + input_.square()), None

2. Select the exact class name under ``model.model_params`` in your existing
   configuration:

.. code-block:: json

   {
       "model": {
           "model_params": {
               "surrogate_gradient": "MyCustomSurrogate",
               "surrogate_params": {"scale": 0.5}
           }
       }
   }

3. Start a new run. No changes to ``get_spike_grad`` or registration are needed.

The lookup finds the class by name in the same module, just as model lookup
finds a class in ``models.py``. For an autograd Function, it binds
``surrogate_params`` to ``forward(ctx, input_, ...)`` and creates the callable
that invokes ``.apply(input_, ...)``. Parameters following ``ctx`` and
``input_`` must accept positional arguments; keyword-only parameters are not
supported by ``.apply``. Omitted values use the defaults declared in
``forward``. Return one gradient per input from ``backward``, using ``None``
for configuration parameters that are not differentiated.

Factories returning a callable and ordinary callable classes are also
supported; their constructors receive ``surrogate_params`` as keyword arguments.
Custom names must be distinct from snnTorch names; collisions raise an error.
Removing a custom class or factory makes its name unavailable in subsequent
runs, without leaving a separate registration entry to remove.
``MyCustomSurrogate`` is a documentation example, not a built-in surrogate;
add the class before selecting its name.

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
* accept ``time_steps`` when used through benchmark configuration; and
* return ``(spikes, membrane)``, each shaped ``(time, batch, classes)``.

For example, this declaration makes ``"MySpikingModel"`` a valid model name:

.. code-block:: python

   class MySpikingModel(torch.nn.Module):
       is_snn = True

       def __init__(self, time_steps=25, output_dim=10):
           super().__init__()
           self.time_steps = time_steps
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
