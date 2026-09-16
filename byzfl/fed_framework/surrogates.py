"""Lookup of built-in and user-defined surrogate gradients by name.

Add custom torch.autograd.Function subclasses, callable classes, or factories
directly to this module and select their names in the configuration.
No registration is required.
"""

import inspect
import sys

import torch
from snntorch import surrogate


def get_spike_grad(surrogate_gradient, surrogate_params):
    """Construct a surrogate callable from its name and keyword parameters."""
    if not isinstance(surrogate_gradient, str) or not surrogate_gradient:
        raise TypeError("surrogate_gradient must be a non-empty string.")
    params = surrogate_params if surrogate_params is not None else {}
    if not isinstance(params, dict):
        raise TypeError("surrogate_params must be a dict or None.")
    factory = getattr(surrogate, surrogate_gradient, None)
    custom_factory = getattr(sys.modules[__name__], surrogate_gradient, None)
    # The lookup helper itself is not a surrogate factory.
    if custom_factory is get_spike_grad:
        custom_factory = None
    if factory is not None and custom_factory is not None:
        raise ValueError(f"Custom surrogate {surrogate_gradient!r} conflicts with a snnTorch name.")
    if custom_factory is not None:
        factory = custom_factory
    if not callable(factory):
        raise ValueError(f"Unknown surrogate gradient: {surrogate_gradient!r}")
    if isinstance(factory, type) and issubclass(factory, torch.autograd.Function):
        # Autograd Function.apply takes positional arguments, not keywords.
        arguments = inspect.signature(factory.forward).bind(None, None, **params)
        arguments.apply_defaults()
        if arguments.kwargs:
            raise TypeError("Surrogate forward parameters must accept positional arguments.")
        forward_params = arguments.args[2:]  # Exclude ctx and the input tensor.
        return lambda input_: factory.apply(input_, *forward_params)
    spike_grad = factory(**params)
    if not callable(spike_grad):
        raise TypeError(f"Surrogate factory {surrogate_gradient!r} must return a callable.")
    return spike_grad
