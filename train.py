import random
import time

import numpy as np
import torch

from server import Server
from compute_cluster import ComputeCluster
from dataset import get_dataloaders
from managers import FileManager

class Train(object):
    """
    Description
    -----------
    Class with implements the train algorithms and stores some stadistics
    of the training

    Parameters
    ----------
        All this parameters should be passed in a dictionary that contains the following keys.
    nb_steps : int 
        Number of steps to do in the Training
    evaluation_delta : int
        How many steps it have to wait to compute accuracy
    evaulate_on_test : bool
        Indicates if will be evaluate on test set
    model_name : str
        Indicates the model to be used
    device : str 
        Name of the device used
    bit_precision : int
        How many bits will be displayed in the accuracy
    learning_rate : float
        Learning rate
    weight_decay : float 
        Weight decay used
    learning_rate_decay : float
        Lerning rate decay used
    batch_size : int
        Batch size used in the train dataloaders
    batch_size_validation: int
        Batch size used in the validation dataloader
    size_train_set: float
        Proportion of data from the train that will be used to train
    dataset_name : str 
        Name of the dataset used
    nb_labels : int
        Number of labels in the dataset
    data_folder : str
        Path of the folder that contains the data
    data_distribution_name : str
        Name of the data distribution
    data_distribution_parameters : dict
        Parameters of the data distribution
    aggregator_info : dict
        Dictionary with the keys "name" and "parameters" defined.
    pre_agg_list : list
        List of dictionaries (one for every pre_agg function)
        where every dictionary have the keys "name" and "parameters" defined.
    nb_workers : int
        Number of workers
    nb_byz : int
        Number of byzantine nodes
    loss_name : str 
        Loss name to be used
    milestones : list 
        List of the milestones, where the learning rate decay should be applied
    learning_rate_decay : float
        Rate decreases over time during training
    momentum : float
        Momentum
    attack_name : str 
        Attack name
    attack_parameters : dict 
        Dictionary with the parameters of the attack where every key is the name 
        of the paramater and their value is the value of the parameter
    attack_optimizer_name : (str, optional)
        Name of the optimizer to be used
    attack_optimizer_parameters : (dict, optional)
        Dictionary with the parameters of the optimizer where every 
        key is the name of the paramater and their value is the value 
        of the parameter
    save_models : (bool)
        If true every delta steps the model will be saved

    Methods
    -------
    """
    def __init__(self, params, dict_params):

        if params["nb_honest"] is not None:
            params["nb_workers"] = params["nb_honest"] + params["nb_byz"]

        file_manager_params = {
            "result_path": params["results_directory"],
            "dataset_name": params["dataset_name"],
            "model_name": params["model_name"],
            "nb_workers": params["nb_workers"],
            "nb_byz": params["nb_byz"],
            "data_distribution_name": params["data_distribution_name"],
            "data_distribution_params": params["data_distribution_parameters"],
            "aggregation_name": params["aggregator_info"]["name"],
            "pre_aggregation_names":  [
                dict['name']
                for dict in params["pre_agg_list"]
            ],
            "attack_name": params["attack_name"],
            "learning_rate": params["learning_rate"],
            "momentum": params["momentum"],
            "weight_decay": params["weight_decay"],
            "learning_rate_decay": params["learning_rate_decay"]
        }

        self.file_manager = FileManager(file_manager_params)
        self.file_manager.save_config_dict(dict_params)

        # Forcing nb_declared = nb_real
        params["aggregator_info"]["parameters"]["nb_byz"] = params["nb_byz"]
        if len(params["pre_agg_list"]) > 0:
            for pre_agg in params["pre_agg_list"]:
                pre_agg["parameters"]["nb_byz"] = params["nb_byz"]

        params_dataloaders = {
            "dataset_name": params["dataset_name"],
            "batch_size": params["batch_size"],
            "batch_size_validation": params["batch_size_validation"],
            "size_train_set": params["size_train_set"],
            "dataset_name": params["dataset_name"],
            "nb_labels": params["nb_labels"],
            "data_folder": params["data_folder"],
            "data_distribution_name": params["data_distribution_name"],
            "data_distribution_parameters": params["data_distribution_parameters"],
            "nb_workers": params["nb_workers"],
            "nb_byz": params["nb_byz"]
        }

        np.random.seed(0)
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        random.seed(0)

        train_dataloaders, validation_dataloader, test_dataloader = get_dataloaders(params_dataloaders)

        self.seed = params["seed"]
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)

        server_params = {
            "model_name": params["model_name"],
            "dataloader": validation_dataloader,
            "test_dataloader": test_dataloader,
            "evaluate_on_test": params["evaluate_on_test"],
            "device": params["device"],
            "learning_rate": params["learning_rate"],
            "weight_decay": params["weight_decay"],
            "milestones": params["milestones"],
            "learning_rate_decay": params["learning_rate_decay"],
            "aggregator_info": params["aggregator_info"],
            "pre_agg_list": params["pre_agg_list"],
            "bit_precision": params["bit_precision"]
        }

        self.server = Server(server_params)

        if params["nb_byz"] == 0:
            params["attack_name"] = "NoAttack"

        compute_cluster_params = {
            "device": params["device"],
            "nb_workers": params["nb_workers"],
            "nb_byz": params["nb_byz"],
            "model_name": params["model_name"],
            "learning_rate": params["learning_rate"],
            "loss_name": params["loss_name"],
            "weight_decay": params["weight_decay"],
            "milestones": params["milestones"],
            "learning_rate_decay": params["learning_rate_decay"],
            "momentum": params["momentum"],
            "attack_name": params["attack_name"],
            "attack_parameters": params["attack_parameters"],
            "attack_optimizer_name": params["attack_optimizer_name"],
            "attack_optimizer_parameters": params["attack_optimizer_parameters"],
            "aggregator_info": params["aggregator_info"],
            "pre_agg_list": params["pre_agg_list"],
            "dataloaders": train_dataloaders,
            "nb_labels": params["nb_labels"]
        }

        self.compute_cluster = ComputeCluster(compute_cluster_params)

        self.compute_cluster.set_model_state(self.server.get_dict_parameters())

        self.steps = params["nb_steps"]
        self.evaluation_delta = params["evaluation_delta"]
        self.evaluate_on_test = params["evaluate_on_test"]
        self.save_models = params["save_models"]
        self.display = params["display_results"]

        #Stored for display results only
        self.agg_name = params["aggregator_info"]["name"]
        self.data_dist_name = params["data_distribution_name"]
        self.attack_name = params["attack_name"]
        self.nb_byz = params["nb_byz"]

        self.accuracy_list = np.array([])
        self.test_accuracy_list = np.array([])
        self.loss_list = np.array([])

        self.use_batch_norm_stats = False
    
    def run_SGD(self):
        """
        Description
        -----------
        Trains the model running SGD for n steps (setting.json)
        """
        start_time = time.time()

        self.server.compute_batch_norm_keys()

        if len(self.server.get_batch_norm_keys()) > 0:
            self.use_batch_norm_stats = True
            self.compute_cluster.compute_batch_norm_keys()
        
        for step in range(0, self.steps):

            if step % self.evaluation_delta == 0:
                accuracy = self.server.compute_validation_accuracy()
                self.accuracy_list = np.append(self.accuracy_list, accuracy)

                if self.evaluate_on_test:
                    test_accuracy = self.server.compute_test_accuracy()
                    self.test_accuracy_list = np.append(
                        self.test_accuracy_list, 
                        test_accuracy
                    )
                    self.file_manager.write_array_in_file(
                        self.test_accuracy_list, 
                        "test_accuracy_seed_" + str(self.seed) + ".txt"
                    )
                
                if self.save_models:
                    self.file_manager.save_state_dict(
                        self.server.get_dict_parameters(),
                        self.seed,
                        step
                    )
            
            #Training
            new_gradients = self.compute_cluster.get_momentum()

            if self.use_batch_norm_stats:
                new_batch_norm_stats = self.compute_cluster.get_batch_norm_stats()

            #Aggregation and update of the global model
            self.server.update_model(new_gradients)

            if self.use_batch_norm_stats:
                self.server.update_batch_norm_stats(new_batch_norm_stats)

            #Broadcasting
            new_parameters = self.server.get_dict_parameters()
            self.compute_cluster.set_model_state(new_parameters)

        accuracy = self.server.compute_validation_accuracy()
        self.accuracy_list = np.append(self.accuracy_list, accuracy)
        self.loss_list = self.compute_cluster.get_loss_list_of_clients()

        self.file_manager.write_array_in_file(
            self.accuracy_list, 
            "train_accuracy_seed_" + str(self.seed) + ".txt"
        )

        for i, loss in enumerate(self.loss_list):
            self.file_manager.save_loss(loss, self.seed, i)

        if self.evaluate_on_test:
            test_accuracy = self.server.compute_test_accuracy()
            self.test_accuracy_list = np.append(self.test_accuracy_list, test_accuracy)
            self.file_manager.write_array_in_file(
                self.test_accuracy_list, 
                "test_accuracy_seed_" + str(self.seed) + ".txt"
            )
        
        if self.save_models:
            self.file_manager.save_state_dict(
                self.server.get_dict_parameters(),
                self.seed,
                self.steps
            )
        
        end_time = time.time()
        execution_time = end_time - start_time

        self.file_manager.write_array_in_file(
            np.array(execution_time),
            "train_time_seed_" + str(self.seed) + ".txt"
        )

        if self.display:
            print("\n")
            print("Agg: "+ self.agg_name)
            print("DataDist: "+ 
                    self.data_dist_name)
            print("Attack: " + self.attack_name)
            print("Nb_byz: " + str(self.nb_byz))
            print("Seed: " + str(self.seed))
            accuracy_str = ", ".join(str(accuracy) for accuracy in self.accuracy_list)
            print("Accuracy list: " + accuracy_str)
            print("\n")