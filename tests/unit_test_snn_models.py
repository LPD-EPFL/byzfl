"""Check SNN temporal outputs, gradients, and independence between batches."""

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import byzfl
from byzfl.benchmark.managers import ParamsManager
from byzfl.fed_framework import models
from byzfl.fed_framework.encoding import TemporalEncoder
from byzfl.fed_framework.model_base_interface import ModelBaseInterface
from byzfl.utils.model_utils import get_model_class


MODEL_CASES = [
    ("fc_snn", (1, 28, 28)),
    ("cnn_mnist_snn", (1, 28, 28)),
    ("cnn_cifar_snn", (3, 32, 32)),
]


@pytest.fixture(scope="module", autouse=True)
def small_cpu_workloads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("ann_name,snn_name", [
    ("fc_mnist", "fc_snn"),
    ("cnn_mnist", "cnn_mnist_snn"),
    ("cnn_cifar", "cnn_cifar_snn"),
])
def test_default_layer_dimensions_match_ann_counterparts(ann_name, snn_name):
    def layer_dimensions(model):
        return [(type(layer), tuple(layer.weight.shape)) for layer in model.modules()
                if isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d))]

    assert layer_dimensions(get_model_class(ann_name)()) == layer_dimensions(get_model_class(snn_name)())


@pytest.mark.parametrize("name,shape", MODEL_CASES)
def test_model_outputs_gradients_and_state_reset(name, shape):
    torch.manual_seed(7)
    model_class = get_model_class(name)
    assert getattr(byzfl, name) is model_class
    assert model_class.is_snn is True
    model = model_class(output_dim=4)
    sequence = torch.randn(2, 3, *shape, requires_grad=True)
    original = sequence.detach().clone()
    output = model(sequence)
    assert isinstance(output, tuple) and len(output) == 2
    spikes, membrane = output
    assert spikes.shape == membrane.shape == (3, 2, 4)
    assert torch.all((spikes == 0) | (spikes == 1))
    assert torch.isfinite(membrane).all()
    (spikes.sum() + membrane.square().mean()).backward()
    assert torch.isfinite(sequence.grad).all() and sequence.grad.abs().sum() > 0
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert next(model.parameters()).grad.abs().sum() > 0
    torch.testing.assert_close(sequence.detach(), original, rtol=0, atol=0)

    # A different batch and sequence length must not influence the next forward.
    model.eval()
    with torch.no_grad():
        model(torch.randn(1, 2, *shape))
        repeated = model(sequence.detach())
    for expected, actual in zip(output, repeated):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    # Resetting the neuron state must also detach the previous autograd graph.
    model.train()
    model.zero_grad(set_to_none=True)
    model(sequence.detach())[1].square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("name,shape", MODEL_CASES)
def test_constant_encoding_matches_explicit_repetition(name, shape):
    model = get_model_class(name)()
    static = torch.randn(2, *shape)
    with torch.no_grad():
        static_output = model(TemporalEncoder(2)(static))
        temporal_output = model(static.unsqueeze(1).expand(-1, 2, -1, -1, -1))
        shorter_output = model(static.unsqueeze(1))
    for expected, actual in zip(static_output, temporal_output):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert shorter_output[0].shape == (1, 2, 10)


def test_fc_model_accepts_encoded_vectors():
    model = models.fc_snn(input_dim=5, hidden_dim=8, output_dim=3)
    static = torch.randn(2, 5)
    with torch.no_grad():
        actual = model(TemporalEncoder(4)(static))
        expected = model(static.unsqueeze(1).expand(-1, 4, -1))
    assert actual[0].shape == (4, 2, 3)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize("name,shape", MODEL_CASES)
def test_constructor_rejects_unknown_parameters(name, shape):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        get_model_class(name)(surrogate_param={"alpha": 2.0})


@pytest.mark.parametrize("shape", [(2, 5), (2, 1, 28, 28), (5,), (2, 1, 1, 1, 1, 1), (0, 2, 5), (2, 0, 5)])
def test_invalid_or_empty_input_is_rejected(shape):
    model = models.fc_snn(input_dim=5, hidden_dim=8)
    with pytest.raises(ValueError, match="SNN inputs"):
        model(torch.zeros(shape))


def test_model_interface_and_configuration_use_real_snn_metadata():
    configuration = ParamsManager({"model": {
        "name": "fc_snn", "encoding": {"time_steps": 2},
        "model_params": {"input_dim": 5, "hidden_dim": 8, "output_dim": 3},
    }}).resolve_model_config()
    interface = ModelBaseInterface(dict(
        model_name=configuration["name"], is_snn=configuration["is_snn"],
        model_params=configuration["model_params"], encoding=configuration["encoding"],
        device="cpu", optimizer_name=None, learning_rate=None, weight_decay=None,
        milestones=None, learning_rate_decay=None,
    ))
    assert interface.is_snn
    assert interface.model(interface.encoder(torch.randn(2, 5)))[0].shape == (2, 2, 3)


@pytest.mark.parametrize("name", ["cnn_mnist_snn", "cnn_cifar_snn"])
def test_learnable_thresholds_receive_gradients(name):
    shape = dict(MODEL_CASES)[name]
    model = get_model_class(name)(threshold=0.5, learn_threshold=True)
    model(torch.randn(2, 2, *shape))[1].square().mean().backward()
    thresholds = [p for key, p in model.named_parameters() if key.endswith("threshold")]
    assert thresholds
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in thresholds)
