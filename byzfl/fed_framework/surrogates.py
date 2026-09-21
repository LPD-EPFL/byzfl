"""Lookup of built-in and user-defined surrogate-gradient factories."""

from snntorch import surrogate


CUSTOM_SURROGATES = {}


def get_spike_grad(surrogate_gradient, surrogate_params):
    """Construct a surrogate callable from its name and keyword parameters."""
    if not isinstance(surrogate_gradient, str) or not surrogate_gradient:
        raise TypeError("surrogate_gradient must be a non-empty string.")
    params = surrogate_params if surrogate_params is not None else {}
    if not isinstance(params, dict):
        raise TypeError("surrogate_params must be a dict or None.")

    custom_factory = CUSTOM_SURROGATES.get(surrogate_gradient)
    snntorch_factory = getattr(surrogate, surrogate_gradient, None)
    if custom_factory is not None and snntorch_factory is not None:
        raise ValueError(f"Custom surrogate {surrogate_gradient!r} conflicts with a snnTorch name.")
    factory = custom_factory if custom_factory is not None else snntorch_factory
    if not callable(factory):
        raise ValueError(f"Unknown surrogate gradient: {surrogate_gradient!r}")

    spike_grad = factory(**params)
    if not callable(spike_grad):
        raise TypeError(f"Surrogate factory {surrogate_gradient!r} must return a callable.")
    return spike_grad
