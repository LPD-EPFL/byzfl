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
            + "d_" + str(params["declared_nb_byz"]) + "_" 
            + params["data_distribution_name"] + "_"
            + str(params["distribution_parameter"]) + "_" 
            + params["aggregation_name"] + "_"
            + "_".join(params["pre_aggregation_names"]) + "_"
            + params["attack_name"] + "_" 
            + "lr_" + str(params["learning_rate"]) + "_" 
            + "mom_" + str(params["momentum"]) + "_" 
            + "wd_" + str(params["weight_decay"]) + "/"
        )
        
        if not os.path.exists(self.files_path):
            try:
                os.makedirs(self.files_path)
            except Exception as e:
                print(e)

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
    
    def save_state_dict(self, state_dict, training_seed, data_dist_seed, step):
        if not os.path.exists(
            self.files_path+"models_tr_seed_" + str(training_seed)
            +"_dd_seed_"+str(data_dist_seed)
        ):
            os.makedirs(self.files_path+"models_tr_seed_" + str(training_seed) 
                        + "_dd_seed_"+str(data_dist_seed))
            
        torch.save(state_dict, self.files_path+"models_tr_seed_" + str(training_seed) 
                   + "_dd_seed_"+str(data_dist_seed)+"/model_step_"+ str(step) +".pth")
    
    def save_loss(self, loss_array, training_seed, data_dist_seed, client_id):
        if not os.path.exists(
            self.files_path+"loss_tr_seed_" + str(training_seed) 
            + "_dd_seed_"+str(data_dist_seed)
        ):
            os.makedirs(self.files_path + "loss_tr_seed_" + str(training_seed) 
                        + "_dd_seed_" + str(data_dist_seed))
            
        file_name = self.files_path + "loss_tr_seed_" + str(training_seed) \
                    + "_dd_seed_" + str(data_dist_seed) \
                    + "/loss_client_" + str(client_id) + ".txt"
        
        np.savetxt(file_name, loss_array, fmt='%.6f', delimiter=",")
    
    def save_accuracy(self, acc_array, training_seed, data_dist_seed, client_id):
        if not os.path.exists(
            self.files_path+"accuracy_tr_seed_" + str(training_seed) 
            + "_dd_seed_"+str(data_dist_seed)
        ):
            os.makedirs(self.files_path+"accuracy_tr_seed_" + str(training_seed) 
                        + "_dd_seed_"+str(data_dist_seed))
            
        file_name = self.files_path+"accuracy_tr_seed_" + str(training_seed) \
                    + "_dd_seed_"+str(data_dist_seed) \
                    + "/accuracy_client_" + str(client_id) + ".txt"
        
        np.savetxt(file_name, acc_array, fmt='%.4f', delimiter=",")




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
            if idx in obj.keys():
                obj = obj[idx]
            else:
                return None
        return obj
    
    def get_flatten_info(self):
        return {
            "training_seed": self.get_training_seed(),
            "device": self.get_device(),
            "nb_workers": self.get_nb_workers(),
            "nb_honest": self.get_nb_honest(),
            "nb_byz": self.get_nb_byz(),
            "declared_nb_byz": self.get_declared_nb_byz(),
            "declared_equal_real": self.get_declared_equal_real(),
            "size_train_set": self.get_size_train_set(),
            "nb_steps": self.get_nb_steps(),
            "evaluation_delta": self.get_evaluation_delta(),
            "evaluate_on_test": self.get_evaluate_on_test(),
            "store_training_accuracy": self.get_store_training_accuracy(),
            "store_training_loss": self.get_store_training_loss(),
            "store_models": self.get_store_models(),
            "batch_size_validation": self.get_batch_size_validation(),
            "data_folder": self.get_data_folder(),
            "results_directory": self.get_results_directory(),
            "model_name": self.get_model_name(),
            "dataset_name": self.get_dataset_name(),
            "nb_labels": self.get_nb_labels(),
            "data_distribution_seed": self.get_data_distribution_seed(),
            "data_distribution_name": self.get_name_data_distribution(),
            "distribution_parameter": self.get_parameter_data_distribution(),
            "loss_name": self.get_loss(),
            "aggregator_info": self.get_aggregator_info(),
            "pre_agg_list": self.get_preaggregators(),
            "batch_norm_momentum": self.get_batch_norm_momentum(),
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
                "training_seed": self.get_training_seed(),
                "device": self.get_device(),
                "nb_workers": self.get_nb_workers(),
                "nb_honest": self.get_nb_honest(),
                "nb_byz": self.get_nb_byz(),
                "declared_nb_byz": self.get_declared_nb_byz(),
                "declared_equal_real": self.get_declared_equal_real(),
                "size_train_set": self.get_size_train_set(),
                "nb_steps": self.get_nb_steps(),
                "evaluation_delta": self.get_evaluation_delta(),
                "evaluate_on_test": self.get_evaluate_on_test(),
                "store_training_accuracy": self.get_store_training_accuracy(),
                "store_training_loss": self.get_store_training_loss(),
                "store_models": self.get_store_models(),
                "batch_size_validation": self.get_batch_size_validation(),
                "data_folder": self.get_data_folder(),
                "results_directory": self.get_results_directory(),
            },

            "model": {
                "name": self.get_model_name(),
                "dataset_name": self.get_dataset_name(),
                "nb_labels": self.get_nb_labels(),
                "data_distribution_seed": self.get_data_distribution_seed(),
                "data_distribution": {
                        "name": self.get_name_data_distribution(),
                        "distribution_parameter": self.get_parameter_data_distribution()
                },
                "loss": self.get_loss(),
            },

            "aggregator": {
                    "name": self.get_aggregator_name(),
                    "parameters": self.get_aggregator_parameters()
            },
        
            "pre_aggregators" : self.get_preaggregators(),

            "server": {
                "batch_norm_momentum": self.get_batch_norm_momentum(),
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

    def get_training_seed(self):
        default = 0
        path = ["general", "training_seed"]
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
    
    def get_declared_nb_byz(self):
        default = 0
        path = ["general", "declared_nb_byz"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_declared_equal_real(self):
        default = False
        path = ["general", "declared_equal_real"]
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

    def get_store_training_accuracy(self):
        default = True
        path = ["general", "store_training_accuracy"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_store_training_loss(self):
        default = True
        path = ["general", "store_training_loss"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_store_models(self):
        default = True
        path = ["general", "store_models"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_batch_size_validation(self):
        default = 100
        path = ["general", "batch_size_validation"]
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

    def get_data_distribution_seed(self):
        default = 0
        path = ["model", "data_distribution_seed"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_name_data_distribution(self):
        default = "iid"
        path = ["model", "data_distribution", "name"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_parameter_data_distribution(self):
        default = None
        path = ["model", "data_distribution", "distribution_parameter"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_loss(self):
        default = "NLLLoss"
        path = ["model", "loss"]
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

    def get_batch_norm_momentum(self):
        default = 0.1
        path = ["server", "batch_norm_momentum"]
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
