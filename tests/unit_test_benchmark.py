import sys
import os
import json

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from byzfl import run_benchmark

class TestBenchmark:

    def test_run_benchmark(self):

        try:
            default_config = {
                "benchmark_config": {
                    "device": "cuda",
                    "training_seed": 0,
                    "nb_training_seeds": 1,
                    "nb_honest_clients": 10,
                    "f": [
                        1
                    ],
                    "tolerated_f": [
                        1
                    ],
                    "filter_non_matching_f_tolerated_f": True,
                    "set_honest_clients_as_clients": False,
                    "size_train_set": 0.8,
                    "data_distribution_seed": 0,
                    "nb_data_distribution_seeds": 1,
                    "data_distribution": [
                        {
                            "name": "gamma_similarity_niid",
                            "distribution_parameter": [
                                0.0
                            ]
                        }
                    ]
                },
                "model": {
                    "name": "cnn_mnist",
                    "dataset_name": "mnist",
                    "nb_labels": 10,
                    "loss": "NLLLoss"
                },
                "aggregator": [
                    {
                        "name": "GeometricMedian",
                        "parameters": {
                            "nu": 0.1,
                            "T": 3
                        }
                    }
                ],
                "pre_aggregators": [
                    {
                        "name": "Clipping",
                        "parameters": {}
                    },
                    {
                        "name": "NNM",
                        "parameters": {}
                    }
                ],
                "server": {
                    "learning_rate": 0.1,
                    "nb_steps": 800,
                    "batch_size_evaluation": 100,
                    "learning_rate_decay": 1.0,
                    "milestones": []
                },
                "honest_nodes": {
                    "momentum": 0.9,
                    "weight_decay": 0.0001,
                    "batch_size": 25
                },
                "attack": [
                    {
                        "name": "SignFlipping",
                        "parameters": {}
                    }
                ],
                "evaluation_and_results": {
                    "evaluation_delta": 50,
                    "evaluate_on_test": True,
                    "store_training_accuracy": True,
                    "store_training_loss": True,
                    "store_models": False,
                    "data_folder": "./data",
                    "results_directory": "./results"
                }
            }
            
            with open('config.json', 'w') as f:
                json.dump(default_config, f, indent=4)
            run_benchmark(1)
            # Look all files have been created
            base_path = os.path.join("results")
            assert os.path.exists(base_path), "Results folder not created"

            experiment_path = os.path.join(base_path, "mnist_cnn_mnist_n_11_f_1_d_1_gamma_similarity_niid_0.0_GeometricMedian_Clipping_NNM_SignFlipping_lr_0.1_mom_0.9_wd_0.0001")
            assert os.path.exists(base_path), "Experiment folder not created"

            # Define expected folder and files
            accuracy_dir = os.path.join(experiment_path, "accuracy_tr_seed_0_dd_seed_0")
            loss_dir = os.path.join(experiment_path, "loss_tr_seed_0_dd_seed_0")

            for i in range(10):
                assert os.path.exists(os.path.join(accuracy_dir, f"accuracy_client_{i}.txt")), f"Missing accuracy_client_{i}.txt"
                assert os.path.exists(os.path.join(loss_dir, f"loss_client_{i}.txt")), f"Missing loss_client_{i}.txt"

            assert os.path.exists(os.path.join(experiment_path, "test_accuracy_tr_seed_0_dd_seed_0.txt"))
            assert os.path.exists(os.path.join(experiment_path, "train_time_tr_seed_0_dd_seed_0.txt"))
            assert os.path.exists(os.path.join(experiment_path, "val_accuracy_tr_seed_0_dd_seed_0.txt"))
            assert os.path.exists(os.path.join(experiment_path, "day.txt"))
            assert os.path.exists(os.path.join(experiment_path, "config.json"))

        except Exception as e:
            pytest.fail(f"run_benchmark raised an unexpected exception: {e}")