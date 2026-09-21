"""Small offline benchmark runs and stable result identities."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")

import numpy as np
from PIL import Image
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from byzfl.benchmark.managers import ParamsManager, get_model_result_name
from byzfl.benchmark.benchmark import eliminate_experiments_done
train = importlib.import_module("byzfl.benchmark.train")
results = importlib.import_module("byzfl.benchmark.evaluate_results")


class TinyMNIST(torch.utils.data.Dataset):
    def __init__(self, root, train, download, transform):
        self.targets = [i % 10 for i in range(20 if train else 4)]
        self.transform = transform
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, index):
        image = Image.fromarray(np.full((28, 28), 20 + index * 7, dtype=np.uint8))
        return self.transform(image), self.targets[index]


def config(tmp_path, encoding="constant"):
    return ParamsManager({
        "benchmark_config": {"nb_steps": 2, "nb_honest_clients": 2, "nb_workers": 2,
            "f": 0, "tolerated_f": 0, "size_train_set": 0.8,
            "data_distribution": {"name": "iid", "distribution_parameter": None},
            "nb_training_seeds": 1, "nb_data_distribution_seeds": 1},
        "model": {"name": "fc_snn", "model_params": {"hidden_dim": 5},
            "encoding": {"type": encoding, "time_steps": 3}, "learning_rate": 0.1},
        "aggregator": {"name": "Average", "parameters": {}}, "pre_aggregators": [],
        "attack": {"name": "SignFlipping", "parameters": {}},
        "honest_clients": {"batch_size": 4},
        "evaluation_and_results": {"results_directory": str(tmp_path), "evaluation_delta": 1,
            "evaluate_on_test": True, "batch_size_evaluation": 3, "store_models": False}
    }).get_data()


@pytest.mark.parametrize("encoding", ["constant", "rate", "latency"])
def test_offline_training_round_trip(tmp_path, monkeypatch, encoding):
    monkeypatch.setattr(train.datasets, "MNIST", TinyMNIST)
    p = config(tmp_path, encoding)
    train.start_training(deepcopy(p))
    folders = [x for x in tmp_path.iterdir() if x.is_dir()]
    assert len(folders) == 1
    directory = folders[0]
    assert get_model_result_name(p) in directory.name
    saved = json.loads((directory / "config.json").read_text())
    assert saved["model"]["encoding"]["type"] == encoding
    curves = {}
    for kind in ("val", "test"):
        curves[kind] = np.loadtxt(directory / f"{kind}_accuracy_tr_seed_0_dd_seed_0.txt", delimiter=",")
        assert curves[kind].shape == (3,)
        assert np.isfinite(curves[kind]).all()
    actual = results._test_accuracy_at_best_validation(
        tmp_path, directory.name, 1, 1, 0, 0, 1
    )
    assert actual == curves["test"][curves["val"].argmax()]
    assert eliminate_experiments_done([p]) == []
    other = deepcopy(p)
    other["model"]["encoding"]["time_steps"] = 4
    assert eliminate_experiments_done([other]) == [other]
    (tmp_path / "config.json").write_text(json.dumps(p))
    results.find_best_hyperparameters(str(tmp_path))
    assert any((tmp_path / "best_hyperparameters").iterdir())
    # Exercise the same result identity in both heatmap readers without rendering.
    monkeypatch.setattr(results.plt, "savefig", lambda *args, **kwargs: None)
    results.test_heatmap(str(tmp_path), str(tmp_path / "plots"))
    results.aggregated_test_heatmap(str(tmp_path), str(tmp_path / "plots"))
    results.plt.close("all")


def test_identity_defaults_order_and_ann_compatibility(tmp_path):
    minimal = {"model": {"name": "fc_snn"}}
    assert get_model_result_name(minimal).startswith("fc_snn_atan_T25_")
    resolved = ParamsManager(minimal).get_data()
    assert get_model_result_name(minimal) == get_model_result_name(resolved)
    model = resolved["model"]
    resolved["model"] = dict(reversed(list(model.items())))
    assert get_model_result_name(minimal) == get_model_result_name(resolved)
    assert get_model_result_name({"model": {"name": "cnn_mnist"}}) == "cnn_mnist"
    original = config(tmp_path)
    changed = deepcopy(original)
    changed["benchmark_config"]["training_seed"] = 8
    changed["model"]["learning_rate"] = 0.9
    assert get_model_result_name(original) == get_model_result_name(changed)


def test_readable_surrogate_and_duration_keep_full_settings_identity(tmp_path):
    p = config(tmp_path)
    p["model"]["model_params"]["surrogate_gradient"] = "fast_sigmoid"
    name = get_model_result_name(p)
    assert name.startswith("fc_snn_fast_sigmoid_T3_")
    assert len(name.rsplit("_", 1)[1]) == 16
    changed = deepcopy(p)
    changed["model"]["model_params"]["surrogate_params"] = {"slope": 10}
    other = get_model_result_name(changed)
    assert other.startswith("fc_snn_fast_sigmoid_T3_")
    assert other != name


@pytest.mark.parametrize("field,value", [
    ("model_params", {"hidden_dim": 7}), ("encoding", {"type": "rate", "time_steps": 3}),
    ("loss", "ce_count_loss"), ("loss_params", {"population_code": True}),
    ("accuracy_name", "accuracy_temporal"),
])
def test_each_snn_setting_changes_identity(tmp_path, field, value):
    p = config(tmp_path)
    changed = deepcopy(p)
    changed["model"][field] = value
    assert get_model_result_name(p) != get_model_result_name(changed)


def test_reader_expands_snn_settings_without_expanding_learning_rate(tmp_path):
    p = config(tmp_path)
    p["model"]["encoding"]["time_steps"] = [3, 4]
    p["model"]["model_params"]["beta"] = [0.9, 0.95]
    p["model"]["learning_rate"] = [0.1, 0.2]
    (tmp_path / "config.json").write_text(json.dumps(p))
    seen = []
    @results._for_each_snn_model
    def read(path, *, _config=None):
        seen.append(_config)
    read(tmp_path)
    assert len(seen) == 4
    assert len({get_model_result_name(c) for c in seen}) == 4
    assert all(c["model"]["learning_rate"] == [0.1, 0.2] for c in seen)
    assert p["model"]["encoding"]["time_steps"] == [3, 4]
