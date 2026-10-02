"""Tests for reporting test accuracy at the best validation checkpoint."""

import importlib
import json
from pathlib import Path
import sys

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
results = importlib.import_module("byzfl.benchmark.evaluate_results")


def write_curve(directory, kind, values, training_seed=3, dd_seed=7):
    path = directory / f"{kind}_accuracy_tr_seed_{training_seed}_dd_seed_{dd_seed}.txt"
    np.savetxt(path, [values], delimiter=",")


@pytest.fixture
def experiment(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps({
        "benchmark_config": {"nb_steps": 10},
    }))
    return directory


def selected_accuracy(directory, training_seeds=1, dd_seeds=1):
    return results._test_accuracy_at_best_validation(
        directory.parent,
        directory.name,
        dd_seeds,
        training_seeds,
        3,
        7,
        4,
    )


@pytest.mark.parametrize("validation,test,expected", [
    ([0.1, 0.9, 0.5, 0.8], [0.95, 0.2, 0.99, 0.3], 0.2),
    ([0.1, 0.9, 0.5, 0.8], [0.95, 0.0, 0.99, 0.3], 0.0),
    ([0.1, 0.9, 0.9, 0.8], [0.95, 0.2, 0.99, 0.3], 0.2),
])
def test_returns_test_accuracy_at_best_validation(
    experiment, validation, test, expected
):
    write_curve(experiment, "val", validation)
    write_curve(experiment, "test", test)
    assert selected_accuracy(experiment) == pytest.approx(expected)


def test_does_not_fallback_when_selected_test_is_unavailable(experiment):
    write_curve(experiment, "val", [0.1, 0.9, 0.5, 0.8])
    write_curve(experiment, "test", [0.95, np.nan, 0.99, 0.3])
    with pytest.raises(ValueError, match="best validation step"):
        selected_accuracy(experiment)


def test_averages_all_runs_before_selecting_checkpoint(experiment):
    curves = {
        (3, 7): ([0.0, 0.9, 0.7, 0.1], [0.0, 0.1, 0.2, 0.3]),
        (4, 7): ([0.0, 0.3, 0.8, 0.1], [0.0, 0.4, 0.8, 0.9]),
        (3, 8): ([0.0, 0.5, 0.9, 0.1], [0.0, 0.2, 0.6, 0.4]),
        (4, 8): ([0.0, 0.7, 0.6, 0.1], [0.0, 0.8, 0.4, 0.7]),
    }
    for (training_seed, dd_seed), (validation, test) in curves.items():
        write_curve(experiment, "val", validation, training_seed, dd_seed)
        write_curve(experiment, "test", test, training_seed, dd_seed)

    # Mean validation is [0.0, 0.6, 0.75, 0.1], selecting index 2.
    # Mean test accuracy there is (0.2 + 0.8 + 0.6 + 0.4) / 4 = 0.5.
    accuracy = selected_accuracy(experiment, training_seeds=2, dd_seeds=2)
    assert accuracy == pytest.approx(0.5)


@pytest.fixture
def benchmark_results(tmp_path):
    config = {
        "benchmark_config": {
            "training_seed": 3,
            "nb_training_seeds": 1,
            "data_distribution_seed": 7,
            "nb_data_distribution_seeds": 1,
            "nb_honest_clients": 2,
            "f": 1,
            "nb_steps": 10,
            "set_honest_clients_as_clients": False,
            "data_distribution": {"name": "iid", "distribution_parameter": 0.0},
        },
        "evaluation_and_results": {"evaluation_delta": 4},
        "model": {
            "name": "cnn_mnist",
            "dataset_name": "mnist",
            "learning_rate": 0.2,
        },
        "honest_clients": {"momentum": 0.0, "weight_decay": 0.0},
        "aggregator": [{"name": "Average"}, {"name": "Median"}],
        "pre_aggregators": [],
        "attack": [{"name": "SignFlipping"}, {"name": "InnerProductManipulation"}],
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    scores = {
        ("Average", "SignFlipping"): 0.6,
        ("Average", "InnerProductManipulation"): 0.4,
        ("Median", "SignFlipping"): 0.5,
        ("Median", "InnerProductManipulation"): 0.7,
    }
    for (aggregator, attack), score in scores.items():
        directory = tmp_path / (
            f"mnist_cnn_mnist_n_3_f_1_d_1_iid_0.0_{aggregator}__{attack}"
            "_lr_0.2_mom_0.0_wd_0.0"
        )
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps(config))
        write_curve(directory, "val", [0.1, 0.2, 0.3, 0.9])
        write_curve(directory, "test", [0.99, 0.98, score, score])

    return tmp_path


@pytest.mark.parametrize("plot_function,expected", [
    ("test_heatmap", [0.4, 0.5]),
    ("aggregated_test_heatmap", [0.5]),
])
def test_heatmaps_use_test_accuracy_at_best_validation(
    benchmark_results, monkeypatch, plot_function, expected
):
    plotted = []
    original_heatmap = results.sns.heatmap

    def capture(table, **kwargs):
        plotted.append(np.asarray(table).copy())
        return original_heatmap(table, **kwargs)

    monkeypatch.setattr(results.sns, "heatmap", capture)
    plot_directory = benchmark_results / "plots"
    getattr(results, plot_function)(str(benchmark_results), str(plot_directory))

    np.testing.assert_allclose([table.item() for table in plotted], expected)
