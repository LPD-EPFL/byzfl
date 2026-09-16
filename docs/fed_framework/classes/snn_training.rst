.. _snn-training-label:

Training and evaluating SNNs
============================

``Client`` and ``Server`` identify spiking models through their class declaration,
not their name. Existing ANN parameters and computation remain unchanged.
For available SNN architectures, see :ref:`models-label`.

Requirements and supported data
-------------------------------

``snntorch`` is a required, unpinned dependency in ``requirements.txt``. Install
the repository requirements before using either ANN or SNN models:

.. code-block:: bash

   pip install -r requirements.txt

The built-in SNN benchmark path uses the existing benchmark's static-image dataset
table and encodes static minibatches after data distribution.
Event-camera datasets such as N-MNIST and event-stream preprocessing are outside
the scope of this integration.

Direct client and server parameters
-----------------------------------

In addition to their usual parameters, both classes accept these SNN fields:

.. code-block:: python

   snn_params = {
       "model_name": "fc_snn",
       "model_params": {"hidden_dim": 100, "beta": 0.95},
       "encoding": {"type": "rate", "time_steps": 25, "encoding_params": {}},
       "accuracy_name": "accuracy_rate",
   }
   # Merge these fields into your usual client/server parameter dictionaries.
   # Clients additionally require:
   client_loss_params = {"loss_name": "ce_rate_loss", "loss_params": {}}

``is_snn`` is optional; when supplied it must be a boolean matching the class.
The encoding defaults to ``constant``. Its duration is passed to the model
constructor; conflicting explicit constructor and encoding durations are rejected.
When encoding duration is omitted, the model's ``time_steps`` is used (25 for
built-in models). Custom models used with an explicit encoding duration must
accept the ``time_steps`` constructor argument.

Data loaders provide static floating-point images or vectors. Encoding runs after
moving each batch to the model's device, during both training and evaluation.
Constant encoding passes the original input through. Rate encoding generates fresh
spikes for each batch use, including evaluation. Latency encoding processes samples
independently because normalized spike times must not depend on other batch members.
Do not also encode inside the dataset transform. See the encoding documentation for
input normalization requirements.

SNN execution uses one device. The framework does not wrap SNN models in
``torch.nn.DataParallel`` because their outputs use ``(time, batch, classes)``
while the default gather operation assumes that dimension zero is the batch.

Built-in losses and accuracy
----------------------------

The loss adapter selects spikes for ``ce_rate_loss``, ``ce_count_loss``,
``mse_count_loss``, ``ce_temporal_loss`` and ``mse_temporal_loss``; it selects
membrane potentials for ``ce_max_membrane_loss`` and ``mse_membrane_loss``.
``loss_params`` supplies keyword arguments to the snnTorch loss constructor.

``accuracy_rate`` (default) and ``accuracy_temporal`` use spikes. Accuracy functions
return a finite scalar fraction between zero and one. Evaluation averages batch
scores weighted by batch size, without rounding them to integer counts. Custom
metrics must therefore be compatible with this reduction; a dataset-wide metric
such as macro precision is not generally equal to an average of batch precisions.
An empty evaluation loader or invalid score raises an error instead of reporting zero.

Custom losses and accuracy functions
------------------------------------

Add an ``nn.Module`` class in ``byzfl/utils/snn_loss.py`` and select its class name
with ``loss_name`` (``model.loss`` in benchmark configuration). No registry change
is needed. Its constructor accepts ``loss_params`` and its ``forward`` receives the
full ``(spikes, membrane)`` tuple and target labels. Return a scalar differentiable
loss. For example:

.. code-block:: python

   class MyMembraneLoss(nn.Module):
       def __init__(self, scale=1.0):
           super().__init__()
           self.scale = scale

       def forward(self, outputs, targets):
           spikes, membrane = outputs
           return self.scale * nn.functional.cross_entropy(membrane.mean(0), targets)

Add a function in ``byzfl/utils/snn_accuracy.py`` and select its name with
``accuracy_name``. It also receives the complete tuple, under ``torch.no_grad()``:

.. code-block:: python

   def my_membrane_accuracy(outputs, targets):
       spikes, membrane = outputs
       predictions = membrane.mean(0).argmax(dim=1)
       return (predictions == targets).float().mean().item()

Use distinct names from the built-in losses, adapters, and lookup helpers.
These custom examples are documentation only; they are not additional built-in metrics.

Benchmark configuration and results
-----------------------------------

Use the same benchmark configuration as for ANN models. A complete configuration
is available at ``docs/fed_framework/configs/snn_mnist.json``:

.. literalinclude:: ../configs/snn_mnist.json
   :language: json

The benchmark distributes static data before clients encode minibatches. It passes
the same encoding and accuracy settings to every client and the server. Constant
encoding keeps the dataset normalization; rate and latency encoding omit it.

SNN result names include ``<model_name>_<surrogate>_T<time_steps>_<identifier>``,
for example ``fc_snn_atan_T25_<identifier>``. An omitted surrogate uses the model
constructor's default name, or ``default`` if no such parameter is declared.
The identifier is the
first 16 hexadecimal characters of a SHA-256 digest of the resolved model name,
constructor parameters, encoding, loss name and parameters, and accuracy selection. JSON
key ordering does not affect it. The full resolved settings remain in each run's
``config.json``; seed-specific filenames keep the existing format. ANN directory
names and file formats remain unchanged.

Lists in SNN settings expand into independent configurations using the existing
benchmark sweep mechanism. Result readers and plots evaluate each SNN configuration
separately, retaining the existing learning-rate, momentum and weight-decay selection
within each configuration. Resume checks use the same identifier as training.

Copy this file to ``config.json`` in the working directory and launch the existing
benchmark entry point:

.. code-block:: python

   from byzfl.benchmark import run_benchmark

   if __name__ == "__main__":
       run_benchmark(nb_jobs=1)

``run_benchmark`` downloads the configured dataset when necessary. Change ``device``
to ``"cuda"`` to use a CUDA device. The example uses one concrete value per field;
lists create sweeps as described in :ref:`federated_learning-label`.

Compatibility summary
---------------------

Existing ANN configuration keys, defaults, model construction, result directory
names, and metric filenames retain their previous behavior. ANN configurations do
not need ``is_snn``, ``model_params``, ``encoding``, ``loss_params``, or
``accuracy_name``. For every model family, test heatmaps now report test accuracy
according to :ref:`checkpoint selection <validation-selected-test-label>`.

The three built-in SNN models support both DSGD and the benchmark's existing FedAvg
path. FedAvg's existing restriction with ``LabelFlipping`` is shared by ANN and SNN
runs. This integration does not change the aggregation or attack algorithms.
