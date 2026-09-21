"""Loss adapters for (spikes, membrane) model outputs."""

import sys

from torch import nn
from snntorch import functional as sf


SNN_LOSS_REGISTRY = {
    "ce_rate_loss": 0,
    "ce_count_loss": 0,
    "mse_count_loss": 0,
    "ce_max_membrane_loss": 1,
    "mse_membrane_loss": 1,
    "ce_temporal_loss": 0,
    "mse_temporal_loss": 0,
}


class SNNLoss(nn.Module):
    """Select the tensor expected by a built-in snnTorch loss."""

    def __init__(self, loss_fn, input_index=0):
        super().__init__()
        self.loss_fn = loss_fn
        self.input_index = input_index

    def forward(self, outputs, targets):
        tensor = outputs[self.input_index] if isinstance(outputs, tuple) else outputs
        return self.loss_fn(tensor, targets)


def create_snn_loss(loss_name, **loss_params):
    """Resolve a built-in loss or a local custom nn.Module class by name.

    Custom classes receive the complete model output in forward().
    """
    if not isinstance(loss_name, str) or not loss_name:
        raise TypeError("loss_name must be a non-empty string.")
    if loss_name in SNN_LOSS_REGISTRY:
        return SNNLoss(getattr(sf, loss_name)(**loss_params), SNN_LOSS_REGISTRY[loss_name])
    loss_class = getattr(sys.modules[__name__], loss_name, None)
    if (not isinstance(loss_class, type) or not issubclass(loss_class, nn.Module)
            or loss_class is SNNLoss):
        raise ValueError(f"Unknown SNN loss: {loss_name!r}")
    return loss_class(**loss_params)
