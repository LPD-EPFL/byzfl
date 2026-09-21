SNN Input Encoding
==================

``TemporalEncoder`` converts static minibatches into inputs for the SNN models.
Distribute static samples first, then encode each batch on its training or
evaluation device before calling the model. Use the same encoder settings for
training, validation, and test data.

.. code-block:: python

   from byzfl.fed_framework.encoding import TemporalEncoder

   encoder = TemporalEncoder(time_steps=25, encoding_type="rate")
   for inputs, labels in client_loader:
       inputs = inputs.to(device)
       spikes, membrane = model(encoder(inputs))

The modes are:

* ``constant``: return the input unchanged. The model repeats it lazily over
  its configured time steps, avoiding a materialized temporal copy.
* ``rate``: call ``snntorch.spikegen.rate`` with the configured parameters.
  A fresh spike sample is drawn on each call; reproducibility depends on the
  PyTorch random seed and call order.
* ``latency``: call ``snntorch.spikegen.latency`` with the configured parameters.
  Each sample is encoded separately, so optional latency normalization does
  not depend on the other samples in the minibatch.

The input must be a floating-point tensor with shape ``(batch, features)`` or
``(batch, channels, height, width)``. Rate and latency outputs insert time after
batch, giving ``(batch, time, ...)``. Already encoded inputs must go directly
to the model rather than through this encoder again.

``encoding_params`` are passed to the selected snnTorch spike generator.
Configure duration only through ``time_steps``; ``num_steps`` must not also
appear in ``encoding_params``. In benchmark JSON, these fields belong under
``model.encoding``. Constant encoding accepts no generator parameters.

Rate and latency inputs are clamped to [0, 1] before encoding. Their image
preprocessing should omit mean/std normalization; constant encoding retains
the normal dataset normalization. ``get_snn_transforms`` in
``byzfl/benchmark/data.py`` prepares these transform chains without modifying
the original transforms.

``load_snn_data`` in that module prepares static training, validation, and test
data from an existing vision dataset entry. Training and validation use
separate dataset instances so their transformations cannot overwrite each
other. It preserves targets for ``DataDistributor`` and does not encode data
during partitioning. These helpers are specific to SNN data preparation.
