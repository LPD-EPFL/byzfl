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
    
    def get_nb_honest(self):
        default = 0
        path = ["benchmark_config", "nb_honest"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_nb_byz(self):
        default = 0
        path = ["benchmark_config", "nb_byz"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_declared_nb_byz(self):
        default = 0
        path = ["benchmark_config", "declared_nb_byz"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_declared_equal_real(self):
        default = False
        path = ["benchmark_config", "declared_equal_real"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_fix_workers_as_honest(self):
        default = False
        path = ["benchmark_config", "fix_workers_as_honest"]
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

    def get_name_data_distribution(self):
        default = "iid"
        path = ["benchmark_config", "data_distribution", "name"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_parameter_data_distribution(self):
        default = None
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
        path = ["aggregator", "parameters"]
        return self._read_object(path)

    # ----------------------------------------------------------------------
    #  Pre-Aggregators
    # ----------------------------------------------------------------------
    def get_preaggregators(self):
        path = ["pre_aggregators"]
        return self._read_object(path)

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
    
    def get_server_weight_decay(self):
        default = 1e-4
        path = ["server", "weight_decay"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_nb_steps(self):
        default = 800
        path = ["server", "nb_steps"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_server_batch_norm_momentum(self):
        default = None
        path = ["server", "batch_norm_momentum"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_server_batch_size_validation(self):
        default = 100
        path = ["server", "batch_size_validation"]
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
        default = 0.99
        path = ["honest_nodes", "momentum"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_honest_nodes_batch_size(self):
        default = 25
        path = ["honest_nodes", "batch_size"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    # ----------------------------------------------------------------------
    #  Attack
    # ----------------------------------------------------------------------

    def get_attack_name(self):
        default = "Infinity"
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
        default = True
        path = ["evaluation_and_results", "store_models"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_data_folder(self):
        default = None
        path = ["evaluation_and_results", "data_folder"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_results_directory(self):
        default = "results"
        path = ["evaluation_and_results", "results_directory"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)