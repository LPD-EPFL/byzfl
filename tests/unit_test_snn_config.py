"""Tests for SNN configuration validation and ANN compatibility."""

from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from byzfl.benchmark.benchmark import generate_all_combinations
from byzfl.benchmark.managers import ParamsManager
from byzfl.fed_framework import models


class SpikingConfigModel:
    is_snn = True

    def __init__(self):
        raise AssertionError("Configuration resolution must not construct a model")


@pytest.fixture(autouse=True)
def register_model(monkeypatch):
    monkeypatch.setattr(models, "SpikingConfigModel", SpikingConfigModel, raising=False)


def snn_config(**fields):
    return {"model": {"name": "SpikingConfigModel", **fields}}


def test_snn_defaults_and_canonical_time_steps():
    resolved = ParamsManager(snn_config()).resolve_model_config()
    assert resolved == {
        "name": "SpikingConfigModel", "is_snn": True, "model_params": {},
        "encoding": {"type": "constant", "time_steps": 25, "encoding_params": {}},
        "loss": "ce_rate_loss", "loss_params": {}, "accuracy_name": "accuracy_rate",
    }


def test_explicit_snn_values_are_preserved_and_resolution_does_not_mutate_input():
    config = snn_config(
        is_snn=True,
        model_params={"beta": 0.95, "surrogate_gradient": "atan", "surrogate_params": {"alpha": 1.2}},
        encoding={"type": "latency", "time_steps": 10, "encoding_params": {"tau": 2.0}},
        loss="ce_temporal_loss", loss_params={"inverse": "negate"},
        accuracy_name="custom_accuracy",
    )
    original = deepcopy(config)
    manager = ParamsManager(config)
    resolved = manager.resolve_model_config()
    assert resolved == original["model"]
    resolved["model_params"]["surrogate_params"]["alpha"] = 4.0
    resolved["encoding"]["encoding_params"]["tau"] = 3.0
    assert config == original
    assert manager.get_time_steps() == 10
    assert "time_steps" not in manager.get_model_params()


@pytest.mark.parametrize("encoding", ["constant", "rate", "latency"])
def test_loss_and_metric_defaults_do_not_depend_on_input_encoding(encoding):
    manager = ParamsManager(snn_config(encoding={"type": encoding}))
    assert manager.get_loss_name() == "ce_rate_loss"
    assert manager.get_accuracy_name() == "accuracy_rate"


def test_ann_serialized_model_and_loss_defaults_are_unchanged():
    expected = {
        "name": "cnn_mnist", "dataset_name": "mnist", "nb_labels": 10,
        "loss": "NLLLoss", "learning_rate": 0.1,
        "learning_rate_decay": 1.0, "milestones": [],
    }
    for config in ({}, {"model": {"name": "cnn_mnist", "is_snn": False}}):
        manager = ParamsManager(config)
        assert manager.get_data()["model"] == expected
        assert manager.get_accuracy_name() is None
        assert not manager.is_snn()


def test_ann_custom_loss_and_ignored_extra_fields_keep_existing_behavior():
    manager = ParamsManager({"model": {
        "name": "cnn_mnist", "loss": "CrossEntropyLoss", "learning_rate": 0.05,
        "encoding": "unused by ANN", "model_params": {"unused": True},
    }})
    serialized = manager.get_data()["model"]
    assert serialized["loss"] == "CrossEntropyLoss"
    assert serialized["learning_rate"] == 0.05
    assert "encoding" not in serialized
    assert "model_params" not in serialized


def test_serialized_snn_config_includes_resolved_values():
    data = ParamsManager(snn_config(encoding={"time_steps": 8})).get_data()
    assert data["model"]["is_snn"] is True
    assert data["model"]["encoding"] == {"type": "constant", "time_steps": 8, "encoding_params": {}}
    assert data["model"]["loss"] == "ce_rate_loss"
    assert data["model"]["dataset_name"] == "mnist"


def test_snn_serialization_keeps_existing_shared_field_defaults():
    data = ParamsManager(snn_config(
        dataset_name=None, nb_labels=None, learning_rate=None,
        learning_rate_decay=None, milestones=None,
    )).get_data()["model"]
    assert data["dataset_name"] == "mnist"
    assert data["nb_labels"] == 10
    assert data["learning_rate"] == 0.1
    assert data["learning_rate_decay"] == 1.0
    assert data["milestones"] == []


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
def test_config_flag_requires_boolean(value):
    with pytest.raises(TypeError, match="model.is_snn"):
        ParamsManager(snn_config(is_snn=value)).resolve_model_config()


@pytest.mark.parametrize("model_name,flag", [("SpikingConfigModel", False), ("cnn_mnist", True)])
def test_config_flag_cannot_override_class(model_name, flag):
    with pytest.raises(ValueError, match="does not match"):
        ParamsManager({"model": {"name": model_name, "is_snn": flag}}).resolve_model_config()


@pytest.mark.parametrize("fields,message", [
    ({"time_steps": 5}, "inside 'model.encoding'"),
    ({"encoding_type": "rate"}, "inside 'model.encoding'"),
    ({"encoding_params": {}}, "inside 'model.encoding'"),
    ({"model_params": {"time_steps": 5}}, "only in 'model.encoding.time_steps'"),
    ({"model_params": {"time_steps": 5}, "encoding": {"time_steps": 5}}, "only in 'model.encoding.time_steps'"),
    ({"encoding": {"time_step": 5}}, "Unknown model.encoding fields"),
    ({"encoding": {"type": "typo"}}, "Unsupported SNN encoding"),
    ({"encoding": {"type": "delta"}}, "Unsupported SNN encoding"),
    ({"time_step": 5}, "Unknown SNN model fields"),
])
def test_ambiguous_or_unknown_snn_configuration_is_rejected(fields, message):
    with pytest.raises(ValueError, match=message):
        ParamsManager(snn_config(**fields)).resolve_model_config()


@pytest.mark.parametrize("time_steps", [0, -1, True, 2.5, "10", [5, 10]])
def test_time_steps_requires_one_positive_integer(time_steps):
    with pytest.raises(ValueError, match="positive integer"):
        ParamsManager(snn_config(encoding={"time_steps": time_steps})).resolve_model_config()


@pytest.mark.parametrize("fields", [
    {"model_params": []}, {"encoding": "rate"}, {"loss_params": []},
    {"encoding": {"encoding_params": []}}, {"encoding": {"type": ["rate", "latency"]}},
    {"loss": ["ce_rate_loss", "ce_temporal_loss"]}, {"accuracy_name": ""},
])
def test_invalid_or_unexpanded_field_types_are_rejected(fields):
    with pytest.raises(TypeError):
        ParamsManager(snn_config(**fields)).resolve_model_config()


def test_sweeps_are_resolved_only_after_existing_expansion():
    grid = {"model": {
        "name": ["cnn_mnist", "SpikingConfigModel"],
        "encoding": {"type": ["constant", "latency"], "time_steps": [5, 10]},
        "model_params": {"beta": [0.9, 0.95]},
    }}
    original = deepcopy(grid)
    manager = ParamsManager(grid)
    assert manager.get_model_name() == ["cnn_mnist", "SpikingConfigModel"]
    assert manager.get_time_steps() == [5, 10]
    with pytest.raises(TypeError, match="expand configuration sweeps"):
        manager.resolve_model_config()
    combinations = generate_all_combinations(grid, ["pre_aggregators", "milestones"])
    assert len(combinations) == 16
    for combination in combinations:
        resolved = ParamsManager(combination).resolve_model_config()
        if resolved["is_snn"]:
            assert resolved["encoding"]["time_steps"] in (5, 10)
            assert resolved["model_params"]["beta"] in (0.9, 0.95)
    assert grid == original


def test_existing_ann_model_sweep_can_still_be_read():
    data = ParamsManager({"model": {"name": ["cnn_mnist", "fc_mnist"]}}).get_data()
    assert data["model"]["name"] == ["cnn_mnist", "fc_mnist"]
    assert data["model"]["loss"] == "NLLLoss"


def test_configuration_resolution_does_not_consume_torch_randomness():
    state = torch.random.get_rng_state().clone()
    ParamsManager(snn_config()).resolve_model_config()
    assert torch.equal(torch.random.get_rng_state(), state)
