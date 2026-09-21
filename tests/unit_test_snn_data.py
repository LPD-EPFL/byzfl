"""Check SNN data splitting before minibatch encoding, without downloads."""

from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image
import pytest
import torch
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from byzfl.benchmark import data
from byzfl.benchmark.managers import ParamsManager
from byzfl.fed_framework.data_distributor import DataDistributor
from byzfl.fed_framework.encoding import TemporalEncoder
from byzfl.fed_framework.models import fc_snn


class SmallVisionDataset(torch.utils.data.Dataset):
    reads = 0

    def __init__(self, root, train, download, transform):
        self.targets = [index % 2 for index in range(10 if train else 4)]
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        type(self).reads += 1
        image = Image.fromarray(np.array([[0, 64], [128, 255]], dtype=np.uint8))
        return self.transform(image), self.targets[index]


@pytest.fixture
def dataset_info(monkeypatch):
    monkeypatch.setattr(data.datasets, "SmallVisionDataset", SmallVisionDataset, raising=False)
    SmallVisionDataset.reads = 0
    train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    return "SmallVisionDataset", train, test


def manager(encoding="rate", train_fraction=0.6):
    return ParamsManager({
        "model": {"name": "fc_snn", "encoding": {"type": encoding, "time_steps": 5}},
        "benchmark_config": {"size_train_set": train_fraction},
        "evaluation_and_results": {"batch_size_evaluation": 2},
    })


@pytest.mark.parametrize("encoding", ["constant", "rate", "latency"])
def test_transform_preparation_preserves_base_chain(dataset_info, encoding):
    base = dataset_info[1]
    before = list(base.transforms)
    result = data.get_snn_transforms(base, encoding)
    assert base.transforms == before
    assert result is not base
    assert result.transforms[0] is before[0]
    assert any(isinstance(t, transforms.Normalize) for t in result.transforms) == (encoding == "constant")


def test_splits_are_disjoint_and_keep_independent_transforms(dataset_info):
    torch.manual_seed(31)
    expected_indices = torch.randperm(10).tolist()
    torch.manual_seed(31)
    train, val, test = data.load_snn_data(manager(), dataset_info)
    assert train.indices == expected_indices[:6]
    assert val.dataset.indices == expected_indices[6:]
    assert train.dataset is not val.dataset.dataset
    assert len(test.dataset) == 4
    assert train.dataset.targets.dtype == torch.long
    assert SmallVisionDataset.reads == 0
    train_image, _ = train.dataset[0]
    val_image, _ = val.dataset.dataset[0]
    test_image, _ = test.dataset[0]
    torch.testing.assert_close(train_image, val_image.flip(-1))
    torch.testing.assert_close(test_image, val_image)
    assert 0 <= val_image.min() <= val_image.max() <= 1


@pytest.mark.parametrize("encoding", ["constant", "rate", "latency"])
def test_distribution_then_encoding_and_snn_backward(dataset_info, encoding):
    params = manager(encoding)
    train, val, test = data.load_snn_data(params, dataset_info)
    random.seed(11)
    loaders = DataDistributor({
        "data_distribution_name": "iid", "nb_honest": 2,
        "data_loader": train, "batch_size": 2,
    }).split_data()
    assert SmallVisionDataset.reads == 0
    assigned = [index for loader in loaders for index in loader.dataset.indices]
    assert sorted(assigned) == sorted(train.indices)
    assert not set(assigned).intersection(val.dataset.indices)
    encoder = TemporalEncoder(5, encoding, {"normalize": True} if encoding == "latency" else {})
    model = fc_snn(input_dim=4, hidden_dim=5, output_dim=2, time_steps=5)
    for loader in [*loaders, val, test]:
        batch, labels = next(iter(loader))
        assert batch.shape == (2, 1, 2, 2)
        encoded = encoder(batch)
        if encoding != "constant":
            assert encoded.shape == (2, 5, 1, 2, 2)
        spikes, membrane = model(encoded)
        assert spikes.shape == membrane.shape == (5, 2, 2)
        model.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(membrane.mean(0), labels).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_no_validation_split(dataset_info):
    train, val, test = data.load_snn_data(manager(train_fraction=1.0), dataset_info)
    assert len(train) == 10 and val is None and len(test.dataset) == 4


def test_ann_cannot_accidentally_use_snn_preparation(dataset_info):
    with pytest.raises(ValueError, match="requires an SNN"):
        data.load_snn_data(ParamsManager({}), dataset_info)
