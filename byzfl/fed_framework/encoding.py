"""Temporal encoding of static minibatches for spiking models."""

import inspect

import torch
from snntorch import spikegen


class TemporalEncoder:
    """Encode a batch of vectors or images, keeping the batch dimension first.

    All encoding modes produce (batch, time, ...) tensors on the input
    device. Inputs to these spike generators are clamped to [0, 1].
    """

    def __init__(self, time_steps=25, encoding_type="constant", encoding_params=None):
        if isinstance(time_steps, bool) or not isinstance(time_steps, int) or time_steps <= 0:
            raise ValueError("time_steps must be a positive integer.")
        if not isinstance(encoding_type, str):
            raise TypeError("encoding_type must be a string.")
        self.encoding_type = encoding_type.lower()
        if self.encoding_type not in ("constant", "rate", "latency"):
            raise ValueError(f"Unsupported SNN encoding: {encoding_type!r}")
        if encoding_params is not None and not isinstance(encoding_params, dict):
            raise TypeError("encoding_params must be a dict or None.")
        self.time_steps = time_steps
        self.encoding_params = dict(encoding_params) if encoding_params is not None else {}
        if "num_steps" in self.encoding_params:
            raise ValueError("Set time_steps instead of encoding_params.num_steps.")
        if self.encoding_type == "constant":
            if self.encoding_params:
                raise ValueError("Constant encoding does not accept encoding_params.")
        else:
            if self.encoding_params.get("time_var_input", False):
                raise ValueError("TemporalEncoder expects static inputs, not time_var_input.")
            # Check keyword names without generating spikes or consuming randomness.
            inspect.signature(getattr(spikegen, self.encoding_type)).bind(
                None, num_steps=self.time_steps, **self.encoding_params
            )

    def __call__(self, batch):
        if not isinstance(batch, torch.Tensor) or not batch.is_floating_point():
            raise TypeError("Encoding requires a floating-point tensor.")
        if batch.dim() not in (2, 4):
            raise ValueError("Encoding requires static (batch, features) or (batch, channels, height, width) inputs.")
        if batch.numel() == 0 or not torch.isfinite(batch).all():
            raise ValueError("Encoding inputs must be non-empty and finite.")
        if self.encoding_type == "constant":
            # Share storage across time instead of copying the static input.
            batch_with_time_axis = batch.unsqueeze(dim=1)
            temporal_shape = (
                batch.shape[0],
                self.time_steps,
                *batch.shape[1:],
            )
            return batch_with_time_axis.expand(*temporal_shape)

        clamped = batch.clamp(0.0, 1.0)
        if self.encoding_type == "latency":
            # Normalize each sample independently of the other batch members.
            return torch.stack([
                spikegen.latency(
                    sample, num_steps=self.time_steps, **self.encoding_params
                )
                for sample in clamped
            ])

        if self.encoding_type == "rate":
            spikes = spikegen.rate(
                clamped, num_steps=self.time_steps, **self.encoding_params
            )
            # snnTorch places time first; the SNN models expect batch first.
            return spikes.movedim(0, 1)

        # The constructor validates the encoding type, so this cannot be reached.
        raise RuntimeError(f"Unsupported encoding type: {self.encoding_type!r}")
