import os
import datetime
import json

import numpy as np
import torch


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
            self.files_path, f"loss_tr_seed_{training_seed}_dd_seed_{data_dist_seed}"
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
            f"accuracy_tr_seed_{training_seed}_dd_seed_{data_dist_seed}"
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
        return {
            "benchmark_config": {
                "device": self.get_device(),
                "training_seed": self.get_training_seed(),
                "nb_training_seeds": self.get_nb_training_seeds(),
                "nb_workers": self.get_nb_workers(),
                "nb_honest_clients": self.get_nb_honest_clients(),
                "f": self.get_f(),
                "tolerated_f": self.get_tolerated_f(),
                "filter_non_matching_f_tolerated_f": self.get_filter_non_matching_f_tolerated_f(),
                "set_honest_clients_as_clients": self.get_set_honest_clients_as_clients(),
                "size_train_set": self.get_size_train_set(),
                "data_distribution_seed": self.get_data_distribution_seed(),
                "nb_data_distribution_seeds": self.get_nb_data_distribution_seeds(),
                "data_distribution": self.get_data_distribution()
            },
            "model": {
                "name": self.get_model_name(),
                "dataset_name": self.get_dataset_name(),
                "nb_labels": self.get_nb_labels(),
                "loss": self.get_loss_name()
            },
            "aggregator": self.get_aggregator_info(),
            "pre_aggregators": self.get_preaggregators(),
            "server": {
                "learning_rate": self.get_server_learning_rate(),
                "nb_steps": self.get_nb_steps(),
                "batch_size_evaluation": self.get_server_batch_size_evaluation(),
                "learning_rate_decay": self.get_server_learning_rate_decay(),
                "milestones": self.get_server_milestones()
            },
            "honest_nodes": {
                "momentum": self.get_honest_nodes_momentum(),
                "weight_decay": self.get_server_weight_decay(),
                "batch_size": self.get_honest_nodes_batch_size()
            },
            "attack": self.get_attack_info(),
            "evaluation_and_results": {
                "evaluation_delta": self.get_evaluation_delta(),
                "evaluate_on_test": self.get_evaluate_on_test(),
                "store_training_accuracy": self.get_store_training_accuracy(),
                "store_training_loss": self.get_store_training_loss(),
                "store_models": self.get_store_models(),
                "data_folder": self.get_data_folder(),
                "results_directory": self.get_results_directory()
            }
        }

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

    def get_filter_non_matching_f_tolerated_f(self):
        default = True
        path = ["benchmark_config", "filter_non_matching_f_tolerated_f"]
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
    #  Server Accessors
    # ----------------------------------------------------------------------
    def get_server_optimizer_name(self):
        default = "SGD"
        path = ["server", "optimizer_name"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_server_learning_rate(self):
        default = 0.1
        path = ["server", "learning_rate"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_nb_steps(self):
        default = 1000
        path = ["server", "nb_steps"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_server_batch_size_evaluation(self):
        default = 128
        path = ["server", "batch_size_evaluation"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_server_learning_rate_decay(self):
        default = 1.0
        path = ["server", "learning_rate_decay"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_server_milestones(self):
        default = []
        path = ["server", "milestones"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    # ----------------------------------------------------------------------
    #  Honest Nodes
    # ----------------------------------------------------------------------
    def get_honest_nodes_momentum(self):
        default = 0.9
        path = ["honest_nodes", "momentum"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_server_weight_decay(self):
        default = 1e-4
        path = ["honest_nodes", "weight_decay"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_honest_nodes_batch_size(self):
        default = 32
        path = ["honest_nodes", "batch_size"]
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

    def get_evaluate_on_test(self):
        default = True
        path = ["evaluation_and_results", "evaluate_on_test"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_store_training_accuracy(self):
        default = True
        path = ["evaluation_and_results", "store_training_accuracy"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_store_training_loss(self):
        default = True
        path = ["evaluation_and_results", "store_training_loss"]
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