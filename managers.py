import os
import datetime
import json

import numpy as np
import torch

class FileManager(object):
    """
    Description
    -----------
    Object whose responsability is deal with the methods and procedures to
    manage the files and store results.
    """
    def __init__(self, params=None):
        
        self.files_path = str(
            params["result_path"] + "/"
            + params["dataset_name"] + "_" 
            + params["model_name"] + "_" 
            "n_" + str(params["nb_workers"]) + "_" 
            + "f_" + str(params["nb_byz"]) + "_" 
            + params["data_distribution_name"] + "_"
            "_".join([
                str(valor) 
                for valor in params["data_distribution_params"].values()
            ]) + "_" 
            + params["aggregation_name"] + "_"
            + "_".join(params["pre_aggregation_names"]) + "_"
            + params["attack_name"] + "_" 
            + "lr_" + str(params["learning_rate"]) + "_" 
            + "mom_" + str(params["momentum"]) + "_" 
            + "wd_" + str(params["weight_decay"]) + "_" 
            + "lr_decay_" + str(params["learning_rate_decay"]) + "/"
        )
        
        if not os.path.exists(self.files_path):
            os.makedirs(self.files_path)

        with open(self.files_path+"day.txt", "w") as file:
            file.write(str(datetime.date.today().strftime("%d_%m_%y")))
        
    def set_experiment_path(self, path):
        self.files_path = path
    
    def get_experiment_path(self):
        return self.files_path
    
    def save_config_dict(self, dict_to_save):
        with open(self.files_path+"settings.json", 'w') as json_file:
            json.dump(dict_to_save, json_file, indent=4, separators=(',', ': '))
    
    def write_array_in_file(self, array, file_name):
        np.savetxt(self.files_path+file_name, [array], fmt='%.4f', delimiter=",")
    
    def save_state_dict(self, state_dict, seed, step):
        if not os.path.exists(self.files_path+"models_seed_"+str(seed)):
            os.makedirs(self.files_path+"models_seed_"+str(seed))
        torch.save(state_dict, self.files_path+"models_seed_"+str(seed)+"/model_step_"+ str(step) +".pth")
    
    def save_loss(self, loss_array, seed, client_id):
        if not os.path.exists(self.files_path+"loss_seed_"+str(seed)):
            os.makedirs(self.files_path+"loss_seed_"+str(seed))
        file_name = self.files_path+"loss_seed_"+str(seed) + "/loss_client_" + str(client_id) + ".txt"
        np.savetxt(file_name, loss_array, fmt='%.10f', delimiter=",")




class ParamsManager(object):
    
    """
    Description
    -----------
    Object whose responsability is manage and store all the parameters.
    """

    def __init__(self, params):
            self.data = params
    
    def _parameter_to_use(self, default, read):
        if read is None:
            return default
        else:
            return read
    
    def _read_object(self, path):
        obj = self.data
        for idx in path:
            obj = obj[idx]
        return obj
    
    def get_flatten_info(self):
        return {
            "seed": self.get_seed(),
            "device": self.get_device(),
            "nb_workers": self.get_nb_workers(),
            "nb_honest": self.get_nb_honest(),
            "nb_byz": self.get_nb_byz(),
            "size_train_set": self.get_size_train_set(),
            "nb_steps": self.get_nb_steps(),
            "evaluation_delta": self.get_evaluation_delta(),
            "evaluate_on_test": self.get_evaluate_on_test(),
            "save_models": self.get_save_models(),
            "data_folder": self.get_data_folder(),
            "results_directory": self.get_results_directory(),
            "bit_precision": self.get_bit_precision(),
            "display_results": self.get_display_results(),
            "model_name": self.get_model_name(),
            "dataset_name": self.get_dataset_name(),
            "nb_labels": self.get_nb_labels(),
            "data_distribution_name": self.get_name_data_distribution(),
            "data_distribution_parameters": self.get_parameters_data_distribution(),
            "loss_name": self.get_loss(),
            "batch_norm": self.get_batch_norm(),
            "aggregator_info": self.get_aggregator_info(),
            "pre_agg_list": self.get_preaggregators(),
            "server_subsampling": self.get_server_subsampling(),
            "batch_size_validation": self.get_batch_size_validation(),
            "momentum": self.get_momentum(),
            "batch_size": self.get_batch_size(),
            "learning_rate": self.get_learning_rate(),
            "weight_decay": self.get_weight_decay(),
            "learning_rate_decay": self.get_learning_rate_decay(),
            "milestones": self.get_milestones(),
            "attack_name": self.get_attack_name(),
            "attack_parameters": self.get_attack_parameters(),
            "attack_optimizer_name": self.get_attack_optimizer_name(),
            "attack_optimizer_parameters": self.get_attack_optimizer_parameters()
        }
    
    def get_data(self):
        return {   
            "general": {
                "seed": self.get_seed(),
                "nb_seeds": self.get_nb_seeds(),
                "device": self.get_device(),
                "nb_workers": self.get_nb_workers(),
                "nb_honest": self.get_nb_honest(),
                "nb_byz": self.get_nb_byz(),
                "size_train_set": self.get_size_train_set(),
                "nb_steps": self.get_nb_steps(),
                "evaluation_delta": self.get_evaluation_delta(),
                "evaluate_on_test": self.get_evaluate_on_test(),
                "save_models": self.get_save_models(),
                "data_folder": self.get_data_folder(),
                "results_directory": self.get_results_directory(),
                "bit_precision": self.get_bit_precision(),
                "display_results": self.get_display_results()
            },

            "model": {
                "name": self.get_model_name(),
                "dataset_name": self.get_dataset_name(),
                "nb_labels": self.get_nb_labels(),
                "data_distribution": {
                        "name": self.get_name_data_distribution(),
                        "parameters": self.get_parameters_data_distribution()
                },
                "loss": self.get_loss(),
                "batch_norm": self.get_batch_norm()
            },

            "aggregator": {
                    "name": self.get_aggregator_name(),
                    "parameters": self.get_aggregator_parameters()
            },
        
            "pre_aggregators" : self.get_preaggregators(),

            "server": {
                "server_subsampling": self.get_server_subsampling(),
                "batch_size_validation": self.get_batch_size_validation(),
                "learning_rate_decay": self.get_learning_rate_decay(),
                "milestones": self.get_milestones()
            },

            "honest_nodes": {
                "momentum": self.get_momentum(),
                "batch_size": self.get_batch_size(),
                "learning_rate": self.get_learning_rate(),
                "weight_decay": self.get_weight_decay()
            },

            "attack": {
                "name": self.get_attack_name(),
                "parameters": self.get_attack_parameters(),
                "attack_optimizer": {
                    "name": self.get_attack_optimizer_name(),
                    "parameters": self.get_attack_optimizer_parameters()
                }
            }
        }
    
    def get_seed(self):
        default = 0
        path = ["general", "seed"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_nb_seeds(self):
        default = 1
        path = ["general", "nb_seeds"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_device(self):
        default = "cpu"
        path = ["general", "device"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_nb_workers(self):
        default = 15
        path = ["general", "nb_workers"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_nb_honest(self):
        default = None
        path = ["general", "nb_honest"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_nb_byz(self):
        default = 0
        path = ["general", "nb_byz"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_size_train_set(self):
        default = 0.8
        path = ["general", "size_train_set"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_nb_steps(self):
        default = 1000
        path = ["general", "nb_steps"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_evaluation_delta(self):
        default = 50
        path = ["general", "evaluation_delta"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_evaluate_on_test(self):
        default = True
        path = ["general", "evaluate_on_test"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_save_models(self):
        default = True
        path = ["general", "save_models"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_data_folder(self):
        default = "./data"
        path = ["general", "data_folder"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_results_directory(self):
        default = "./results"
        path = ["general", "results_directory"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_bit_precision(self):
        default = 6
        path = ["general", "bit_precision"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_display_results(self):
        default = True
        path = ["general", "display_results"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
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
        path = ["model", "nb_labels"]
        return self._read_object(path)
    
    def get_name_data_distribution(self):
        default = "iid"
        path = ["model", "data_distribution", "name"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_parameters_data_distribution(self):
        default = "iid"
        path = ["model", "data_distribution", "parameters"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_loss(self):
        default = "NLLLoss"
        path = ["model", "loss"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_batch_norm(self):
        default = False
        path = ["model", "batch_norm"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
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
        path = ["aggregator", "parameters"]
        return self._read_object(path)
    
    def get_preaggregators(self):
        path = ["pre_aggregators"]
        return self._read_object(path)
    
    def get_server_subsampling(self):
        path = ["server", "server_subsampling"]
        return self._read_object(path)
    
    def get_batch_size_validation(self):
        default = 100
        path = ["server", "batch_size_validation"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_learning_rate_decay(self):
        default = 5000
        path = ["server", "learning_rate_decay"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_momentum(self):
        default = 0.99
        path = ["honest_nodes", "momentum"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_batch_size(self):
        default = 25
        path = ["honest_nodes", "batch_size"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_learning_rate(self):
        default = 0.1
        path = ["honest_nodes", "learning_rate"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_weight_decay(self):
        default = 0
        path = ["honest_nodes", "weight_decay"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_milestones(self):
        default = 200
        path = ["server", "milestones"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_attack_name(self):
        default = "no_attack"
        path = ["attack", "name"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_attack_parameters(self):
        default = "NoAttack"
        path = ["attack", "parameters"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_attack_optimizer_name(self):
        default = None
        if "attack_optimizer" in self.data["attack"]:
            if "name" in self.data["attack"]["attack_optimizer"]:
                path = ["attack", "attack_optimizer", "name"]
                read = self._read_object(path)
                return self._parameter_to_use(default, read)
        return default
    
    def get_attack_optimizer_parameters(self):
        default = {}
        if "attack_optimizer" in self.data["attack"]:
            if "parameters" in self.data["attack"]["attack_optimizer"]:
                path = ["attack", "attack_optimizer", "parameters"]
                read = self._read_object(path)
                return self._parameter_to_use(default, read)
        return default