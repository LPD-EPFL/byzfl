"""Check minibatch encoding layout, parameters, and randomness."""

from pathlib import Path
import sys

import pytest
import torch
from snntorch import spikegen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from byzfl.fed_framework.encoding import TemporalEncoder


@pytest.mark.parametrize("shape", [(2, 4), (2, 1, 2, 2)])
@pytest.mark.parametrize("encoding,params", [
    ("rate", {"gain": 0.8}), ("latency", {"normalize": True, "linear": True}),
    ("latency", {"normalize": True}),
])
def test_batch_encoding_matches_snntorch_without_changing_input(shape, encoding, params):
    batch = torch.linspace(-0.1, 1.1, 8).reshape(shape)
    original = batch.clone()
    encoder = TemporalEncoder(5, encoding, params)
    torch.manual_seed(12)
    if encoding == "latency":
        expected = torch.stack([spikegen.latency(sample.clamp(0, 1), num_steps=5, **params)
                                for sample in batch])
    else:
        expected = spikegen.rate(batch.clamp(0, 1), num_steps=5, **params).movedim(0, 1)
    torch.manual_seed(12)
    actual = encoder(batch)
    assert actual.shape == (2, 5, *shape[1:])
    assert actual.device == batch.device
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(batch, original, rtol=0, atol=0)


def test_constant_preserves_normalized_data_and_random_state():
    batch = torch.tensor([[-2., 0.5, 3.]])
    state = torch.random.get_rng_state().clone()
    assert TemporalEncoder(3, "constant")(batch) is batch
    assert torch.equal(state, torch.random.get_rng_state())


def test_encoder_construction_does_not_draw_spikes_or_mutate_parameters():
    params = {"gain": 0.5}
    state = torch.random.get_rng_state().clone()
    encoder = TemporalEncoder(3, "RATE", params)
    params["gain"] = 0
    assert encoder.encoding_params == {"gain": 0.5}
    assert torch.equal(state, torch.random.get_rng_state())


def test_rate_encoding_is_repeatable_with_seed_and_redrawn_between_calls():
    encoder = TemporalEncoder(8, "rate")
    batch = torch.full((4, 20), 0.5)
    torch.manual_seed(41)
    first = encoder(batch)
    second = encoder(batch)
    assert not torch.equal(first, second)
    torch.manual_seed(41)
    assert torch.equal(first, encoder(batch))


def test_latency_normalization_is_independent_of_batch_members():
    encoder = TemporalEncoder(10, "latency", {"normalize": True})
    image = torch.tensor([[0.5, 0.9]])
    batch = torch.cat([image, torch.tensor([[0.1, 0.2]])])
    torch.testing.assert_close(encoder(batch)[0], encoder(image)[0], rtol=0, atol=0)


@pytest.mark.parametrize("time_steps", [0, -1, True, 1.5, "2"])
def test_invalid_time_steps(time_steps):
    with pytest.raises(ValueError, match="positive integer"):
        TemporalEncoder(time_steps)


@pytest.mark.parametrize("kwargs,error", [
    ({"encoding_type": "typo"}, ValueError),
    ({"encoding_type": None}, TypeError),
    ({"encoding_params": []}, TypeError),
    ({"encoding_type": "rate", "encoding_params": {"num_steps": 5}}, ValueError),
    ({"encoding_type": "rate", "encoding_params": {"time_var_input": True}}, ValueError),
    ({"encoding_type": "rate", "encoding_params": {"gaim": 2}}, TypeError),
    ({"encoding_type": "constant", "encoding_params": {"gain": 2}}, ValueError),
])
def test_invalid_encoding_settings_fail_during_construction(kwargs, error):
    with pytest.raises(error):
        TemporalEncoder(**kwargs)


@pytest.mark.parametrize("batch", [
    torch.zeros(2, 3, 4), torch.empty(0, 4), torch.tensor([[float("nan")]]),
    torch.tensor([[float("inf")]]),
])
def test_invalid_batch_is_rejected(batch):
    with pytest.raises(ValueError):
        TemporalEncoder(3, "rate")(batch)


@pytest.mark.parametrize("batch", [None, [[0.5]], torch.tensor([[1]])])
def test_input_must_already_be_a_float_tensor(batch):
    with pytest.raises(TypeError, match="floating-point tensor"):
        TemporalEncoder()(batch)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_encoding_on_cuda_keeps_device():
    batch = torch.rand(2, 1, 2, 2, device="cuda")
    for encoding, params in (("constant", {}), ("rate", {}), ("latency", {"normalize": True})):
        assert TemporalEncoder(5, encoding, params)(batch).device == batch.device
