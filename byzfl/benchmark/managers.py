import os
import datetime
import json
import hashlib
import inspect
from copy import deepcopy

import numpy as np
import torch

from byzfl.utils.model_utils import get_model_class, is_snn_model


class FileManager:
    """
    Description
    -----------
    Manages the creation of directories and files to store results.
    """

    def __init__(self, params=None):
        self.files_path = (
            f"{params['result_path']}/"
            f"{params['dataset_name']}_{params['model_name']}_"
            f"n_{params['nb_workers']}_"
            f"f_{params['nb_byz']}_"
            f"d_{params['declared_nb_byz']}_"
            f"{params['data_distribution_name']}_"
            f"{params['distribution_parameter']}_"
            f"{params['aggregation_name']}_"
            f"{'_'.join(params['pre_aggregation_names'])}_"
            f"{params['attack_name']}_"
            f"lr_{params['learning_rate']}_"
            f"mom_{params['momentum']}_"
            f"wd_{params['weight_decay']}/"
        )
        os.makedirs(self.files_path, exist_ok=True)

        with open(os.path.join(self.files_path, "day.txt"), "w") as file:
            file.write(datetime.date.today().strftime("%d_%m_%y"))

    def set_experiment_path(self, path):
        """
        Set the base path for the experiment files.
        """
        self.files_path = path

    def get_experiment_path(self):
        """
        Get the current experiment path.
        """
        return self.files_path

    def save_config_dict(self, dict_to_save):
        """
        Save a configuration dictionary as a JSON file.
        """
        config_path = os.path.join(self.files_path, "config.json")
        with open(config_path, "w") as json_file:
            json.dump(dict_to_save, json_file, indent=4, separators=(",", ": "))

    def write_array_in_file(self, array, file_name):
        """
        Write a single array to a file.
        """
        file_path = os.path.join(self.files_path, file_name)
        np.savetxt(file_path, [array], fmt="%.4f", delimiter=",")

    def save_state_dict(self, state_dict, training_seed, data_dist_seed, step):
        """
        Save a model's state dictionary under a directory structured by seed values.
        """
        model_dir = os.path.join(
            self.files_path, f"models_tr_seed_{training_seed}_dd_seed_{data_dist_seed}"
        )
        os.makedirs(model_dir, exist_ok=True)

        file_path = os.path.join(model_dir, f"model_step_{step}.pth")
        torch.save(state_dict, file_path)

    def save_loss(self, loss_array, training_seed, data_dist_seed, client_id):
        """
        Save a loss array for a specific client and seed values.
        """
        loss_dir = os.path.join(
            self.files_path, f"train_loss_tr_seed_{training_seed}_dd_seed_{data_dist_seed}"
        )
        os.makedirs(loss_dir, exist_ok=True)

        file_path = os.path.join(loss_dir, f"loss_client_{client_id}.txt")
        np.savetxt(file_path, loss_array, fmt="%.6f", delimiter=",")

    def save_accuracy(self, acc_array, training_seed, data_dist_seed, client_id):
        """
        Save an accuracy array for a specific client and seed values.
        """
        acc_dir = os.path.join(
            self.files_path,
            f"train_accuracy_tr_seed_{training_seed}_dd_seed_{data_dist_seed}"
        )
        os.makedirs(acc_dir, exist_ok=True)

        file_path = os.path.join(acc_dir, f"accuracy_client_{client_id}.txt")
        np.savetxt(file_path, acc_array, fmt="%.4f", delimiter=",")




class ParamsManager(object):
    """
    Description
    -----------
    Object whose responsibility is to manage and store all the parameters
    from the JSON structure.
    """

    def __init__(self, params):
        self.data = params

    def _parameter_to_use(self, default, read):
        if read is None:
            return default
        else:
            return read

    def _read_object(self, path):
        """
        Safely traverse the nested dictionary `self.data` using the list of keys in `path`.
        Returns None if a key doesn't exist.
        """
        obj = self.data
        for p in path:
            if isinstance(obj, dict) and p in obj.keys():
                obj = obj[p]
            else:
                return None
        return obj

    def get_data(self):
        data = {
            "benchmark_config": {
                "device": self.get_device(),
                "training_seed": self.get_training_seed(),
                "nb_training_seeds": self.get_nb_training_seeds(),
                "nb_workers": self.get_nb_workers(),
                "nb_honest_clients": self.get_nb_honest_clients(),
                "f": self.get_f(),
                "tolerated_f": self.get_tolerated_f(),
                "set_honest_clients_as_clients": self.get_set_honest_clients_as_clients(),
                "size_train_set": self.get_size_train_set(),
                "data_distribution_seed": self.get_data_distribution_seed(),
                "nb_data_distribution_seeds": self.get_nb_data_distribution_seeds(),
                "data_distribution": self.get_data_distribution(),
                "training_algorithm": self.get_training_algorithm(),
                "nb_steps": self.get_nb_steps()
            },
            "model": {
                "name": self.get_model_name(),
                "dataset_name": self.get_dataset_name(),
                "nb_labels": self.get_nb_labels(),
                "loss": self.get_loss_name(),
                "learning_rate": self.get_learning_rate(),
                "learning_rate_decay": self.get_learning_rate_decay(),
                "milestones": self.get_milestones()
            },
            "aggregator": self.get_aggregator_info(),
            "pre_aggregators": self.get_preaggregators(),
            "honest_clients": {
                "momentum": self.get_honest_clients_momentum(),
                "weight_decay": self.get_honest_clients_weight_decay(),
                "batch_size": self.get_honest_clients_batch_size()
            },
            "attack": self.get_attack_info(),
            "evaluation_and_results": {
                "evaluation_delta": self.get_evaluation_delta(),
                "batch_size_evaluation": self.get_batch_size_evaluation(),
                "evaluate_on_test": self.get_evaluate_on_test(),
                "store_per_client_metrics": self.get_store_per_client_metrics(),
                "store_models": self.get_store_models(),
                "data_folder": self.get_data_folder(),
                "results_directory": self.get_results_directory()
            }
        }
        # Preserve ANN output keys and allow unexpanded ANN model lists.
        if isinstance(self.get_model_name(), str) and self.is_snn():
            resolved = self.resolve_model_config()
            # Retain shared fields from the getters so their default
            # handling is not overwritten by raw configuration values such as None.
            for name in ("is_snn", "model_params", "encoding", "loss", "loss_params", "accuracy_name"):
                data["model"][name] = resolved[name]
        return data

    # ----------------------------------------------------------------------
    #  Benchmark Config
    # ----------------------------------------------------------------------

    def get_device(self):
        default = "cpu"
        path = ["benchmark_config", "device"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_training_seed(self):
        default = 0
        path = ["benchmark_config", "training_seed"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_nb_training_seeds(self):
        default = 1
        path = ["benchmark_config", "nb_training_seeds"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_nb_workers(self):
        default = 1
        path = ["benchmark_config", "nb_workers"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_nb_honest_clients(self):
        default = 0
        path = ["benchmark_config", "nb_honest_clients"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_f(self):
        default = 0
        path = ["benchmark_config", "f"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_tolerated_f(self):
        default = self.get_f()
        path = ["benchmark_config", "tolerated_f"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_set_honest_clients_as_clients(self):
        default = False
        path = ["benchmark_config", "set_honest_clients_as_clients"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_size_train_set(self):
        default = 0.8
        path = ["benchmark_config", "size_train_set"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_data_distribution_seed(self):
        default = 0
        path = ["benchmark_config", "data_distribution_seed"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_nb_data_distribution_seeds(self):
        default = 1
        path = ["benchmark_config", "nb_data_distribution_seeds"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_data_distribution(self):
        default = {
                "name": "iid",
                "distribution_parameter": 1.0
        }
        path = ["benchmark_config", "data_distribution"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_name_data_distribution(self):
        default = "iid"
        path = ["benchmark_config", "data_distribution", "name"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_parameter_data_distribution(self):
        default = 1.0
        path = ["benchmark_config", "data_distribution", "distribution_parameter"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_training_algorithm(self):
        default = {
            "name": "DSGD",
            "parameters": {}
        }
        path = ["benchmark_config", "training_algorithm"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_training_algorithm_name(self):
        default = "DSGD"
        path = ["benchmark_config", "training_algorithm", "name"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_training_algorithm_parameters(self):
        default = {}
        path = ["benchmark_config", "training_algorithm", "parameters"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_nb_steps(self):
        default = 1000
        path = ["benchmark_config", "nb_steps"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    # ----------------------------------------------------------------------
    #  Model
    # ----------------------------------------------------------------------
    def get_model_name(self):
        default = "cnn_mnist"
        path = ["model", "name"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_dataset_name(self):
        default = "mnist"
        path = ["model", "dataset_name"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_nb_labels(self):
        default = 10
        path = ["model", "nb_labels"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_loss_name(self):
        default = "NLLLoss"
        path = ["model", "loss"]
        read = self._read_object(path)
        # Defer model-list resolution until after sweep expansion.
        if read is None and isinstance(self.get_model_name(), str) and self.is_snn():
            default = "ce_rate_loss"
        return self._parameter_to_use(default, read)
    
    def get_optimizer_name(self):
        default = "SGD"
        path = ["model", "optimizer_name"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_learning_rate(self):
        default = 0.1
        path = ["model", "learning_rate"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_learning_rate_decay(self):
        default = 1.0
        path = ["model", "learning_rate_decay"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_milestones(self):
        default = []
        path = ["model", "milestones"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    # ----------------------------------------------------------------------
    #  Aggregator
    # ----------------------------------------------------------------------
    def get_aggregator_info(self):
        default = {"name": "Average", "parameters": {}}
        path = ["aggregator"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_aggregator_name(self):
        default = "average"
        path = ["aggregator", "name"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_aggregator_parameters(self):
        default = {}
        path = ["aggregator", "parameters"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    # ----------------------------------------------------------------------
    #  Pre-Aggregators
    # ----------------------------------------------------------------------
    def get_preaggregators(self):
        default = []
        path = ["pre_aggregators"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    # ----------------------------------------------------------------------
    #  Honest Nodes
    # ----------------------------------------------------------------------
    def get_honest_clients_momentum(self):
        default = 0.9
        path = ["honest_clients", "momentum"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_honest_clients_weight_decay(self):
        default = 1e-4
        path = ["honest_clients", "weight_decay"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_honest_clients_batch_size(self):
        default = 32
        path = ["honest_clients", "batch_size"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    # ----------------------------------------------------------------------
    #  Attack
    # ----------------------------------------------------------------------

    def get_attack_info(self):
        default = {"name": "NoAttack", "parameters": {}}
        path = ["attack"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_attack_name(self):
        default = "NoAttack"
        path = ["attack", "name"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_attack_parameters(self):
        default = {}
        path = ["attack", "parameters"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    # ----------------------------------------------------------------------
    #  Evaluation and Results Accessors
    # ----------------------------------------------------------------------
    def get_evaluation_delta(self):
        default = 50
        path = ["evaluation_and_results", "evaluation_delta"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_batch_size_evaluation(self):
        default = 128
        path = ["evaluation_and_results", "batch_size_evaluation"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_evaluate_on_test(self):
        default = True
        path = ["evaluation_and_results", "evaluate_on_test"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_store_per_client_metrics(self):
        default = True
        path = ["evaluation_and_results", "store_per_client_metrics"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_store_models(self):
        default = False
        path = ["evaluation_and_results", "store_models"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_data_folder(self):
        default = "./data"
        path = ["evaluation_and_results", "data_folder"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_results_directory(self):
        default = "./results"
        path = ["evaluation_and_results", "results_directory"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    # ----------------------------------------------------------------------
    #  SNN Properties
    # ----------------------------------------------------------------------

    def is_snn(self):
        """Read the model declaration and check an optional configuration flag."""
        is_snn = is_snn_model(get_model_class(self.get_model_name()))
        model = self._read_object(["model"])
        if isinstance(model, dict) and "is_snn" in model:
            if not isinstance(model["is_snn"], bool):
                raise TypeError("Configuration 'model.is_snn' must be a bool.")
            if model["is_snn"] != is_snn:
                raise ValueError("Configuration 'model.is_snn' does not match the model class declaration.")
        return is_snn

    def get_encoding_type(self):
        """Get SNN encoding type (constant, rate, latency)."""
        val = self._read_object(["model", "encoding", "type"])
        return self._parameter_to_use("constant", val)

    def get_time_steps(self):
        """Get SNN time steps from the canonical encoding configuration."""
        val = self._read_object(["model", "encoding", "time_steps"])
        return self._parameter_to_use(25, val)

    def get_encoding_params(self):
        """Get SNN encoding params."""
        val = self._read_object(["model", "encoding", "encoding_params"])
        return self._parameter_to_use({}, val)

    def get_model_params(self):
        """Get SNN custom model params."""
        val = self._read_object(["model", "model_params"])
        return self._parameter_to_use({}, val)

    def get_loss_params(self):
        """Get SNN custom loss params."""
        val = self._read_object(["model", "loss_params"])
        return self._parameter_to_use({}, val)

    def get_accuracy_name(self):
        """Get SNN accuracy metric name."""
        path = ["model", "accuracy_name"]
        read = self._read_object(path)
        if read is not None:
            return read
        # Input encoding does not determine the output metric.
        if self.is_snn():
            return "accuracy_rate"
        return None

    def _validate_snn_params(self):
        """Validate one concrete SNN configuration without constructing a model."""
        if not self.is_snn():
            return
        model = self._read_object(["model"]) or {}
        known_fields = {
            "name", "dataset_name", "nb_labels", "loss", "learning_rate",
            "learning_rate_decay", "milestones", "optimizer_name", "is_snn",
            "model_params", "encoding", "loss_params", "accuracy_name",
            "time_steps", "encoding_type", "encoding_params",
        }
        unknown = set(model) - known_fields
        if unknown:
            raise ValueError(f"Unknown SNN model fields: {', '.join(sorted(unknown))}")
        for name in ("model_params", "encoding", "loss_params"):
            if model.get(name) is not None and not isinstance(model[name], dict):
                raise TypeError(f"Configuration 'model.{name}' must be a dict.")

        for name in ("time_steps", "encoding_type", "encoding_params"):
            if name in model:
                raise ValueError(f"Put '{name}' inside 'model.encoding', not directly in 'model'.")
        if "time_steps" in self.get_model_params():
            raise ValueError("Configure time_steps only in 'model.encoding.time_steps', not 'model.model_params'.")

        encoding = model.get("encoding") or {}
        unknown = set(encoding) - {"type", "time_steps", "encoding_params"}
        if unknown:
            raise ValueError(f"Unknown model.encoding fields: {', '.join(sorted(unknown))}")
        encoding_type = self.get_encoding_type()
        if not isinstance(encoding_type, str):
            raise TypeError("Configuration 'model.encoding.type' must be a string; expand sweeps first.")
        if encoding_type.lower() not in ("constant", "rate", "latency"):
            raise ValueError(f"Unsupported SNN encoding: {encoding_type!r}")
        time_steps = self.get_time_steps()
        if isinstance(time_steps, bool) or not isinstance(time_steps, int) or time_steps <= 0:
            raise ValueError("Configuration 'model.encoding.time_steps' must be a positive integer; expand sweeps first.")
        if not isinstance(self.get_encoding_params(), dict):
            raise TypeError("Configuration 'model.encoding.encoding_params' must be a dict.")
        for name, value in (("loss", self.get_loss_name()), ("accuracy_name", self.get_accuracy_name())):
            if not isinstance(value, str) or not value:
                raise TypeError(f"Configuration 'model.{name}' must be a non-empty string; expand sweeps first.")

    def resolve_model_config(self):
        """Return a model configuration with its SNN settings resolved after expansion.

        The time_steps setting belongs to the encoder. Models infer the sequence
        length from their temporal inputs.
        """
        model = self._read_object(["model"])
        if model is None:
            model = {}
        if not isinstance(model, dict):
            raise TypeError("Configuration 'model' must be a dict; expand sweeps first.")
        is_snn = self.is_snn()
        resolved = deepcopy(model)
        resolved.update(name=self.get_model_name(), is_snn=is_snn)
        if is_snn:
            self._validate_snn_params()
            resolved.update(
                model_params=deepcopy(self.get_model_params()),
                encoding={
                    "type": self.get_encoding_type().lower(),
                    "time_steps": self.get_time_steps(),
                    "encoding_params": deepcopy(self.get_encoding_params()),
                },
                loss=self.get_loss_name(),
                loss_params=deepcopy(self.get_loss_params()),
                accuracy_name=self.get_accuracy_name(),
            )
        return resolved


def get_model_result_name(params):
    """Keep ANN paths unchanged and distinguish concrete SNN configurations."""
    manager = ParamsManager(params)
    if not manager.is_snn():
        return manager.get_model_name()
    model = manager.resolve_model_config()
    identity = {key: model[key] for key in (
        "name", "model_params", "encoding", "loss", "loss_params", "accuracy_name"
    )}
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
    surrogate = model["model_params"].get("surrogate_gradient")
    if surrogate is None:
        parameter = inspect.signature(get_model_class(model["name"])).parameters.get("surrogate_gradient")
        surrogate = parameter.default if parameter is not None else "default"
        if surrogate is inspect.Parameter.empty:
            surrogate = "default"
    # Keep custom surrogate names safe as a single path component.
    surrogate = "".join(char if char.isalnum() or char in "_-" else "_" for char in str(surrogate))
    return f"{model['name']}_{surrogate}_T{model['encoding']['time_steps']}_{digest}"
