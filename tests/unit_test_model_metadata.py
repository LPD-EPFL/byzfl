"""Model-family metadata must not change existing ANN initialization."""

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from byzfl.fed_framework import models
from byzfl.fed_framework.model_base_interface import ModelBaseInterface
from byzfl.utils.model_utils import get_model_class, is_snn_model


class SpikingToy(torch.nn.Module):
    is_snn = True

    def __init__(self, input_dim=3):
        super().__init__()
        self.layer = torch.nn.Linear(input_dim, 2)


@pytest.fixture
def registered_models(monkeypatch):
    monkeypatch.setattr(models, "SpikingToy", SpikingToy, raising=False)
    monkeypatch.setattr(models, "ann_with_snn_in_its_name", models.cnn_mnist, raising=False)


def interface_params(model_name="cnn_mnist", **extra):
    params = dict(model_name=model_name, device="cpu", optimizer_name=None,
                  learning_rate=None, weight_decay=None, milestones=None,
                  learning_rate_decay=None)
    params.update(extra)
    return params


def test_model_metadata_does_not_depend_on_name(registered_models):
    assert not is_snn_model(get_model_class("ann_with_snn_in_its_name"))
    assert is_snn_model(get_model_class("SpikingToy"))


@pytest.mark.parametrize("value", ["true", "false", 0, 1, None])
def test_class_declaration_requires_an_actual_boolean(value):
    class InvalidModel:
        is_snn = value
    with pytest.raises(TypeError, match="attribute 'is_snn'"):
        is_snn_model(InvalidModel)


def test_unknown_model_is_not_silently_an_ann():
    with pytest.raises(AttributeError):
        get_model_class("unknown_model")


def test_model_sweep_must_be_expanded_before_lookup():
    with pytest.raises(TypeError, match="expand configuration sweeps"):
        get_model_class(["cnn_mnist"])


def test_interface_uses_metadata_and_passes_snn_constructor_arguments(registered_models):
    interface = ModelBaseInterface(interface_params("SpikingToy", model_params={"input_dim": 5}))
    assert interface.is_snn is True
    assert interface.model.layer.in_features == 5
    with pytest.raises(AttributeError):
        interface.is_snn = False


@pytest.mark.parametrize("model_name,flag", [("SpikingToy", True), ("cnn_mnist", False)])
def test_matching_configuration_assertion_is_accepted(registered_models, model_name, flag):
    assert ModelBaseInterface(interface_params(model_name, is_snn=flag)).is_snn is flag


@pytest.mark.parametrize("model_name,flag", [("SpikingToy", False), ("cnn_mnist", True)])
def test_configuration_cannot_override_the_model(registered_models, model_name, flag):
    with pytest.raises(ValueError, match="does not match"):
        ModelBaseInterface(interface_params(model_name, is_snn=flag))


@pytest.mark.parametrize("value", ["true", "false", 0, 1, None])
def test_interface_flag_is_not_coerced(value):
    with pytest.raises(TypeError, match="'is_snn' must be a bool"):
        ModelBaseInterface(interface_params(is_snn=value))


def test_snn_constructor_parameters_must_be_a_dictionary(registered_models):
    with pytest.raises(TypeError, match="'model_params' must be a dict"):
        ModelBaseInterface(interface_params("SpikingToy", model_params=[]))


def test_ann_initialization_and_rng_are_unchanged():
    torch.manual_seed(17)
    expected = models.cnn_mnist()
    expected_next_random = torch.rand(4)
    torch.manual_seed(17)
    # Original ByzFL ignored this optional field for ANN constructors.
    interface = ModelBaseInterface(interface_params(model_params={"unused": 42}))
    assert interface.is_snn is False
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(interface.model.state_dict()[name], value, rtol=0, atol=0)
    torch.testing.assert_close(torch.rand(4), expected_next_random, rtol=0, atol=0)


def test_only_ann_models_use_existing_data_parallel_path(registered_models, monkeypatch):
    wrapped = []

    def fake_parallel(model):
        wrapped.append(model)
        return model

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.nn, "DataParallel", fake_parallel)
    monkeypatch.setattr(torch.nn.Module, "to", lambda self, device: self)
    ann = ModelBaseInterface(interface_params("ann_with_snn_in_its_name", device="cuda"))
    ModelBaseInterface(interface_params("SpikingToy", device="cuda"))
    assert wrapped == [ann.model]
