"""Verify surrogate derivatives and reject invalid factory configuration."""

from pathlib import Path
import sys

import pytest
import torch
from snntorch import surrogate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from byzfl.fed_framework.surrogates import CUSTOM_SURROGATES, get_spike_grad
from byzfl.fed_framework import models
from byzfl.benchmark.managers import ParamsManager


def my_custom_surrogate(scale=1.0):
    def custom_gradient(input_, grad_input, spikes):
        return grad_input * scale / (1 + input_.square())

    return surrogate.custom_surrogate(custom_gradient)


@pytest.fixture
def custom_factory(monkeypatch):
    monkeypatch.setitem(CUSTOM_SURROGATES, "MyCustomSurrogate", my_custom_surrogate)


def test_custom_factory_is_discovered_and_uses_configured_derivative(custom_factory):
    config = ParamsManager({"model": {
        "name": "fc_snn", "model_params": {
            "surrogate_gradient": "MyCustomSurrogate", "surrogate_params": {"scale": 0.5},
        },
    }}).resolve_model_config()["model_params"]
    values = torch.tensor([-2., 0., 2.], requires_grad=True)
    spikes = get_spike_grad(**config)(values)
    torch.testing.assert_close(spikes, torch.tensor([0., 0., 1.]))
    spikes.backward(torch.tensor([1., 2., 3.]))
    torch.testing.assert_close(values.grad, torch.tensor([0.1, 1.0, 0.3]))

    torch.manual_seed(17)
    model = models.fc_snn(input_dim=3, hidden_dim=4, output_dim=2, **config)
    inputs = torch.randn(2, 2, 3, requires_grad=True)
    model(inputs)[1].square().sum().backward()
    assert torch.isfinite(inputs.grad).all() and inputs.grad.abs().sum() > 0
    assert model.fc1.weight.grad.abs().sum() > 0


def test_removing_custom_factory_leaves_no_registration(monkeypatch, custom_factory):
    assert callable(get_spike_grad("MyCustomSurrogate", None))
    with monkeypatch.context() as context:
        context.delitem(CUSTOM_SURROGATES, "MyCustomSurrogate")
        with pytest.raises(ValueError, match="Unknown surrogate"):
            get_spike_grad("MyCustomSurrogate", {})
    assert callable(get_spike_grad("MyCustomSurrogate", None))


def test_custom_name_cannot_shadow_snntorch(monkeypatch):
    monkeypatch.setitem(CUSTOM_SURROGATES, "atan", my_custom_surrogate)
    with pytest.raises(ValueError, match="conflicts"):
        get_spike_grad("atan", {})


def test_custom_factory_rejects_misspelled_parameter(custom_factory):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        get_spike_grad("MyCustomSurrogate", {"scle": 0.5})


def test_custom_factory_defaults_are_used(custom_factory):
    values = torch.tensor([-1., 0., 1.], requires_grad=True)
    get_spike_grad("MyCustomSurrogate", None)(values).sum().backward()
    torch.testing.assert_close(values.grad, torch.tensor([0.5, 1., 0.5]))


def test_custom_callable_class_remains_supported(monkeypatch):
    class CallableSurrogate:
        def __init__(self, scale=1.0):
            self.scale = scale

        def __call__(self, input_):
            return my_custom_surrogate(self.scale)(input_)

    monkeypatch.setitem(CUSTOM_SURROGATES, "CallableSurrogate", CallableSurrogate)
    values = torch.tensor([0.], requires_grad=True)
    get_spike_grad("CallableSurrogate", {"scale": 0.5})(values).sum().backward()
    torch.testing.assert_close(values.grad, torch.tensor([0.5]))


def test_custom_factory_must_return_callable(monkeypatch):
    monkeypatch.setitem(CUSTOM_SURROGATES, "InvalidSurrogate", lambda: None)
    with pytest.raises(TypeError, match="must return a callable"):
        get_spike_grad("InvalidSurrogate", {})


@pytest.mark.parametrize("name,params", [("atan", {"alpha": 1.2}), ("fast_sigmoid", {"slope": 5}), ("triangular", {"threshold": 0.7})])
def test_snntorch_factory_parameters_are_forwarded(name, params):
    actual_input = torch.tensor([-.5, 0., .5], requires_grad=True)
    reference_input = actual_input.detach().clone().requires_grad_()
    actual = get_spike_grad(name, params)(actual_input)
    reference = getattr(surrogate, name)(**params)(reference_input)
    actual.sum().backward()
    reference.sum().backward()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(actual_input.grad, reference_input.grad, rtol=0, atol=0)


def test_default_surrogate_parameters():
    assert callable(get_spike_grad("atan", None))


@pytest.mark.parametrize("name", [None, "", 1])
def test_invalid_surrogate_name_type(name):
    with pytest.raises(TypeError, match="non-empty string"):
        get_spike_grad(name, {})


@pytest.mark.parametrize("name", ["unknown_surrogate", "box", "rectangular", "tri", "get_spike_grad"])
def test_unknown_surrogate_name(name):
    with pytest.raises(ValueError, match="Unknown surrogate"):
        get_spike_grad(name, {})


def test_invalid_surrogate_parameter_container():
    with pytest.raises(TypeError, match="surrogate_params"):
        get_spike_grad("atan", [])


def test_misspelled_surrogate_parameter():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        get_spike_grad("atan", {"alhpa": 1.2})
