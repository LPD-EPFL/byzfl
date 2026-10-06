"""Accuracy functions accepting the complete (spikes, membrane) output."""

import math
import sys

from snntorch import functional as sf


def accuracy_rate(outputs, targets):
    """Classify using the spike count over time."""
    return sf.accuracy_rate(outputs[0], targets)


def accuracy_temporal(outputs, targets):
    """Classify using the first spike time."""
    return sf.accuracy_temporal(outputs[0], targets)


def get_snn_accuracy(name):
    """Find a built-in adapter or a custom function defined in this module."""
    if not isinstance(name, str) or not name:
        raise TypeError("accuracy_name must be a non-empty string.")
    function = getattr(sys.modules[__name__], name, None)
    if not callable(function) or function in (get_snn_accuracy, validate_accuracy):
        raise ValueError(f"Unknown SNN accuracy: {name!r}")
    return function


def validate_accuracy(value):
    """Require a finite scalar fraction so invalid scores cannot be recorded."""
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("SNN accuracy must be a finite scalar between 0 and 1.")
    return value
