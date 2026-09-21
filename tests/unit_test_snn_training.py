"""Client/server integration with temporal inputs and tuple outputs."""
from pathlib import Path
import sys

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from snntorch import functional as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from byzfl.fed_framework.client import Client
from byzfl.fed_framework.server import Server
from byzfl.utils import snn_loss, snn_accuracy


def params(**extra):
    loader = DataLoader(TensorDataset(torch.rand(5, 4), torch.tensor([0, 1, 0, 1, 0])), batch_size=3)
    result = dict(model_name="fc_snn", device="cpu", model_params=dict(input_dim=4, hidden_dim=6, output_dim=2),
                  encoding=dict(type="constant", time_steps=3), loss_name="ce_rate_loss",
                  LabelFlipping=False, nb_labels=2, momentum=0.0, training_dataloader=loader,
                  store_per_client_metrics=True, optimizer_name="SGD", optimizer_params={},
                  learning_rate=0.1, weight_decay=0.0, milestones=[], learning_rate_decay=1.0,
                  test_loader=loader, validation_loader=loader,
                  aggregator_info={"name": "Average", "parameters": {}}, pre_agg_list=[])
    result.update(extra)
    return result


@pytest.mark.parametrize("encoding", ["constant", "rate", "latency"])
def test_training_and_evaluation(encoding):
    p = params(encoding=dict(type=encoding, time_steps=3))
    client, server = Client(p), Server(p)
    assert client.model.time_steps == client.encoder.time_steps == 3
    assert torch.isfinite(torch.tensor(client.compute_gradients()))
    assert torch.isfinite(client.get_flat_gradients()).all()
    assert 0 <= client.get_train_accuracy()[0] <= 1
    assert torch.isfinite(torch.tensor(client.compute_model_update(2)))
    server.set_model_state(client.get_dict_parameters())
    assert 0 <= server.compute_test_accuracy() <= 1
    assert 0 <= server.compute_validation_accuracy() <= 1


def test_label_flipping_encodes_only_once():
    client = Client(params(LabelFlipping=True))
    calls = []
    encoder = client.encoder
    def record(inputs):
        calls.append(inputs)
        return encoder(inputs)
    client.encoder = record
    client.compute_gradients()
    assert len(calls) == 1
    assert torch.isfinite(client.get_flat_flipped_gradients()).all()


@pytest.mark.parametrize("name,index", list(snn_loss.SNN_LOSS_REGISTRY.items()))
def test_builtin_loss_selects_expected_tensor(name, index):
    spikes = torch.zeros(3, 2, 2, requires_grad=True)
    membrane = torch.rand(3, 2, 2, requires_grad=True)
    targets = torch.tensor([0, 1])
    outputs = (spikes, membrane)
    actual = snn_loss.create_snn_loss(name)(outputs, targets)
    expected = getattr(sf, name)()(outputs[index], targets)
    torch.testing.assert_close(actual, expected)


def test_custom_loss_and_accuracy_receive_tuple(monkeypatch):
    seen = []
    class MyLoss(torch.nn.Module):
        def __init__(self, scale=1):
            super().__init__()
            self.scale = scale
        def forward(self, outputs, targets):
            assert isinstance(outputs, tuple)
            seen.append("loss")
            return self.scale * torch.nn.functional.cross_entropy(outputs[1].sum(0), targets)
    def my_accuracy(outputs, targets):
        assert isinstance(outputs, tuple)
        assert not torch.is_grad_enabled()
        seen.append("accuracy")
        return 0.2 if len(targets) == 3 else 0.7
    monkeypatch.setattr(snn_loss, "MyLoss", MyLoss, raising=False)
    monkeypatch.setattr(snn_accuracy, "my_accuracy", my_accuracy, raising=False)
    p = params(loss_name="MyLoss", loss_params={"scale": 2}, accuracy_name="my_accuracy")
    client, server = Client(p), Server(p)
    client.compute_gradients()
    assert client.criterion.scale == 2
    assert client.get_train_accuracy() == [0.2]
    assert server.compute_test_accuracy() == pytest.approx(0.4)
    assert seen.count("loss") == 1 and seen.count("accuracy") == 3


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_accuracy_rejected(value):
    server = Server(params())
    server.accuracy_fn = lambda outputs, targets: value
    with pytest.raises(ValueError, match="finite scalar"):
        server.compute_test_accuracy()


def test_empty_evaluation_rejected():
    server = Server(params())
    empty = DataLoader(TensorDataset(torch.empty(0, 4), torch.empty(0, dtype=torch.long)))
    with pytest.raises(ValueError, match="empty"):
        server._compute_accuracy(empty)


def test_conflicting_duration_rejected():
    p = params()
    p["model_params"]["time_steps"] = 7
    with pytest.raises(ValueError, match="must agree"):
        Client(p)


@pytest.mark.parametrize("constructor", [Client, Server])
def test_model_metadata_assertion_forwarded(constructor):
    with pytest.raises(ValueError, match="does not match"):
        constructor(params(is_snn=False))


def test_ann_client_matches_original_loss_gradients_and_rng():
    loader = DataLoader(TensorDataset(torch.rand(3, 1, 28, 28), torch.tensor([0, 1, 2])), batch_size=3)
    p = params(model_name="fc_mnist", loss_name="CrossEntropyLoss", training_dataloader=loader,
               test_loader=loader, model_params={"ignored": True}, encoding={"ignored": True})
    client = Client(p)
    inputs, targets = next(iter(loader))
    client.model.zero_grad()
    outputs = client.model(inputs)
    expected_loss = torch.nn.functional.cross_entropy(outputs, targets)
    expected_loss.backward()
    expected_grad = client.get_flat_gradients().clone()
    expected_accuracy = (outputs.argmax(1) == targets).float().mean().item()
    rng = torch.random.get_rng_state()
    actual = client.compute_gradients()
    assert actual == expected_loss.item()
    torch.testing.assert_close(client.get_flat_gradients(), expected_grad, rtol=0, atol=0)
    assert client.get_train_accuracy()[0] == pytest.approx(expected_accuracy)
    assert torch.equal(torch.random.get_rng_state(), rng)
    server = Server(p)
    server.set_model_state(client.get_dict_parameters())
    assert server.compute_test_accuracy() == pytest.approx(expected_accuracy)
